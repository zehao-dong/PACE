import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Encoder, SemanticEmbedding, PositionalEmbedding, TokenTypeEmbedding

""" Transformer Encoder """
class GnnLayer(nn.Module):
    def __init__(self, config, device=None):
        super(GnnLayer, self).__init__()
        # GNN encode layer
        self.gnn_layer = nn.Linear(config.indim, config.outdim, bias=False)
        self.act1 = F.tanh
        self.device = device

    def forward(self,x,adj):
        b_size = x.shape[0]
        num_node = x.shape[1] 
        diag_mat = torch.stack([(torch.ones(num_node,1)).to(self._get_device())]*b_size,dim=0) 
        deg_mat = torch.sum(adj,dim=2,keepdim=True) + diag_mat
        message = self.gnn_layer(x)
        message = torch.matmul(adj,message) + message
        message = torch.div(message,deg_mat)
        return message

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device


class GraphEncoder(nn.Module):
    def __init__(self, config):
        super(GraphEncoder, self).__init__()
        # Forward Transformers
        self.encoder_f = Encoder(config)

    def forward(self, x, mask, mask_):
        h_f, hs_f, attns_f = self.encoder_f(x, mask)
        return h_f

    @staticmethod
    def get_embeddings(h_x,l):
        h_x = h_x.cpu()
        b,_,d = h_x.shape
        embed = torch.zeros(b,d)
        for i in range(l.size()[0]):
            embed[i,:] = h_x[i,l[i]-1,:]
        return embed
    #def get_embeddings(h_x):
        #h_x = h_x.cpu()
        #return h_x[:, -1]

class CLSHead(nn.Module):
    def __init__(self, config, init_weights=None):
        super(CLSHead, self).__init__()
        self.layer_1 = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_2 = nn.Linear(config.d_model, config.n_vocab)
        if init_weights is not None:
            self.layer_2.weight = init_weights

    def forward(self, x):
        x = self.dropout(torch.tanh(self.layer_1(x)))
        return F.log_softmax(self.layer_2(x), dim=-1)

class PairWiseLearning(nn.Module):
    def __init__(self, config):
        super(PairWiseLearning, self).__init__()
        # Shared Embedding Layer
        self.opEmb = SemanticEmbedding(config.graph_encoder)
        self.gnn_layer = GnnLayer(config.gnn)
        self.dropout_op = nn.Dropout(p=config.dropout)

        # 2 GraphEncoder for X and Y
        self.graph_encoder = GraphEncoder(config.graph_encoder)

        # Cross Attention between X and Y
        self.segEmb = TokenTypeEmbedding(config.cross_attention)
        self.dropout_seg = nn.Dropout(p=config.dropout)
        self.cross_attention = Encoder(config.cross_attention)

        self.cls = CLSHead(config.cls, init_weights=self.opEmb.w2e.weight if config.tied_weights else None)

    def forward(self, X, maskX, maskX_, Y, maskY, maskY_, maskXY, adjX, adjY):
        # Shared Embeddings
        emb_x = self.dropout_op(self.opEmb(X))
        emb_y = self.dropout_op(self.opEmb(Y))

        emb_x = self.gnn_layer(emb_x,adjX) # added
        emb_y = self.gnn_layer(emb_y,adjY) # added

        segX = torch.zeros_like(X).long()
        segY = torch.ones_like(Y).long()
        seg_x = self.dropout_seg(self.segEmb(segX))
        seg_y = self.dropout_seg(self.segEmb(segY))

        h_x = self.graph_encoder(emb_x, maskX, maskX_)
        h_y = self.graph_encoder(emb_y, maskY, maskY_)
        """
            Shape: Batch Size, Length (with Pad), Feature Dim (forward) + Feature Dim (backward)
            *HINT: X1 X2 X3 [PAD] [PAD] Y1 Y2 Y3 [PAD] [PAD]
        """
        h_ = torch.cat([h_x, h_y], dim=1)
        s_ = torch.cat([seg_x, seg_y], dim=1)
        h, _, _ = self.cross_attention(h_ + s_, maskXY)

        return self.cls(h)