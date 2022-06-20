import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .transformer import Encoder, SemanticEmbedding, PositionalEmbedding, TokenTypeEmbedding

class ClabelEmbedding(nn.Module):
    def __init__(self, config):
        super(ClabelEmbedding, self).__init__()
        self.d_model = int(config.d_model) # no / 2
        #self.w2e = nn.Linear(config.maxL, self.d_model)
        self.c2e = nn.Embedding(config.maxL, self.d_model)

    def forward(self, x):
        #return self.w2e(x) * math.sqrt(self.d_model)
        return self.c2e(x) 
    
class Embedding1(nn.Module):
    def __init__(self, config):
        super(Embedding1, self).__init__()

        self.w2e = SemanticEmbedding(config)
        #self.p2e = PositionalEmbedding(config)
        #self.t2e = TokenTypeEmbedding(config)

        self.dropout = nn.Dropout(p = config.dropout)

    def forward(self, input_ids, position_ids = None, token_type_ids = None):
        #if position_ids is None:
        #    batch_size, length = input_ids.size()
        #    with torch.no_grad():
        #        position_ids = torch.arange(0, length).repeat(batch_size, 1)
        #    if torch.cuda.is_available():
        #        position_ids = position_ids.cuda(device=input_ids.device)
        
        #if token_type_ids is None:
            #token_type_ids = torch.zeros_like(input_ids)

        embeddings = self.w2e(input_ids)
        return self.dropout(embeddings)
    
""" Transformer Encoder """
class GnnLayer(nn.Module):
    def __init__(self, config, scale=0.001, device=None):
        super(GnnLayer, self).__init__()
        # GNN encode layer
        self.gnn_layer = nn.Linear(config.indim, config.outdim, bias=False)
        self.act1 = F.tanh
        self.device = device
        self.scale = scale

    def forward(self,x,adj):
        b_size = x.shape[0]
        num_node = x.shape[1] 
        diag_mat = torch.stack([(torch.ones(num_node,1)).to(self._get_device())]*b_size,dim=0) 
        deg_mat = torch.sum(adj,dim=2,keepdim=True) + diag_mat
        message = self.gnn_layer(x)
        message = torch.matmul(adj,message) + message
        message = torch.div(message,deg_mat)
        message = F.normalize(message,dim=-1) * self.scale 
        return message

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

class CLEmbedding(nn.Module):
    def __init__(self, config):
        super(CLEmbedding, self).__init__()

        p2e = torch.zeros(config.maxL, config.d_model)
        position = torch.arange(0.0, config.maxL).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, config.d_model, 2) * (- math.log(10000.0) / config.d_model))
        p2e[:, 0::2] = torch.sin(position * div_term)
        p2e[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('p2e', p2e)

    def forward(self, x):
        shp = x.size()
        with torch.no_grad():
            emb = torch.index_select(self.p2e, 0, x.view(-1)).view(shp + (-1,))
        return emb
    
class GraphEncoder(nn.Module):
    def __init__(self, config):
        super(GraphEncoder, self).__init__()
        # Forward Transformers
        self.encoder_f = Encoder(config)

    def forward(self, x, mask, mask_):
        h_f, hs_f, attns_f = self.encoder_f(x, mask)
        return h_f

    @staticmethod
    def get_embeddings(h_x):
        h_x = h_x.cpu()
        return h_x[:, -1]
    
    @staticmethod
    def get_embeddings2(h_x):
        h_x = h_x.cpu()
        return h_x

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
        #self.posEmb = ClabelEmbedding(config.graph_encoder) ########### changed
        self.posEmb = CLEmbedding(config.graph_encoder) ##### swift
        self.gnn_layer = GnnLayer(config.gnn)
        self.dropout_op = nn.Dropout(p=config.dropout)

        # 2 GraphEncoder for X and Y
        self.graph_encoder = GraphEncoder(config.graph_encoder)

        # Cross Attention between X and Y
        self.segEmb = TokenTypeEmbedding(config.cross_attention)
        self.dropout_seg = nn.Dropout(p=config.dropout)
        self.cross_attention = Encoder(config.cross_attention)

        self.cls = CLSHead(config.cls, init_weights=self.opEmb.w2e.weight if config.tied_weights else None)

    def forward(self, X, maskX, maskX_, Y, maskY, maskY_, maskXY, adjX, adjY, clabelX, clabelY):
        # Shared Embeddings
        emb_x = self.dropout_op(self.opEmb(X))
        emb_y = self.dropout_op(self.opEmb(Y))
        
        clemb_x = self.dropout_op(self.posEmb(clabelX))
        clemb_y = self.dropout_op(self.posEmb(clabelY))
        
        clemb_x = self.gnn_layer(clemb_x,adjX) # added
        clemb_y = self.gnn_layer(clemb_y,adjY) # added
        
        cuda_device = adjX.get_device()
        length_mask_x = torch.ones_like(maskX).to(cuda_device)
        length_mask_y = torch.ones_like(maskY).to(cuda_device)
        
        segX = torch.zeros_like(X).long()
        segY = torch.ones_like(Y).long()
        seg_x = self.dropout_seg(self.segEmb(segX))
        seg_y = self.dropout_seg(self.segEmb(segY))

        h_x = self.graph_encoder(emb_x+clemb_x, length_mask_x,length_mask_x)
        h_y = self.graph_encoder(emb_y+clemb_y, length_mask_y,length_mask_y)
        #h_x = self.graph_encoder(emb_x, length_mask_x,length_mask_x)
        #h_y = self.graph_encoder(emb_y, length_mask_y,length_mask_y)
        """
            Shape: Batch Size, Length (with Pad), Feature Dim (forward) + Feature Dim (backward)
            *HINT: X1 X2 X3 [PAD] [PAD] Y1 Y2 Y3 [PAD] [PAD]
        """
        h_ = torch.cat([h_x, h_y], dim=1)
        s_ = torch.cat([seg_x, seg_y], dim=1)
        h, _, _ = self.cross_attention(h_ + s_, maskXY)

        return self.cls(h)