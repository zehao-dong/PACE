import math
import random
import copy
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import igraph
import pdb
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.functional import *
from util_bn import longest_path
from torch.nn import LayerNorm

def multi_head_attention_func(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None                    # type: Optional[Tensor]
                                 ):
    
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask.to(torch.uint8), float('-inf'))
            #print(attn_output_weights)
        else:
            attn_output_weights += attn_mask

    
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.uint8),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    #print(attn_output_weights)
    attn_output_weights = softmax(
        attn_output_weights, dim=-1)
    #print(attn_output_weights)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
    #print(attn_output_weights)
    attn_output = torch.bmm(attn_output_weights, v)
    #print(attn_output)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    #print(attn_output)
    
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        #self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiHeadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask= None,
                need_weights= True, attn_mask= None):
        
        if not self._qkv_same_embed_dim:
            return multi_head_attention_func(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_func(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        
class TransformerDecoder_layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5):
        super(TransformerDecoder_layer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoder_layer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask= None):
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # memory mask to target masjk
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=tgt_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _get_activation_fn(self,activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoder_layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",layer_norm_eps=1e-5):
        super(TransformerEncoder_layer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoder_layer, self).__setstate__(state)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """ 
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    def _get_activation_fn(self,activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
        
        
class Attention_out_computation(nn.Module):
    def __init__(self, d_model, nhead, out_dim, dropout=0.1, activation="relu",layer_norm_eps=1e-5):
        super(Attention_out_computation, self).__init__()
        self.self_attn = MultiHeadAttention(out_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, out_dim)
        self.activation = self._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(Attention_out_computation, self).__setstate__(state)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """ 
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src = self.linear1(src)
        out = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        return src
    
    def _get_activation_fn(self,activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
        
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        return output
    def _get_clones(self,module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        return output
    def _get_clones(self,module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class GNNposEncoding(nn.Module):
    def __init__(self, ninp, dropout=0.1, max_n=20,device=None):
        super(GNNposEncoding, self).__init__()
        self.ninp = ninp # size of the position embedding
        self.max_n = max_n # maximum position
        self.dropout = dropout
        if dropout > 0.0001:
            self.droplayer = nn.Dropout(p=dropout)
        self.W1 = nn.Parameter(torch.zeros(2*max_n, 2*ninp))
        self.W2 = nn.Parameter(torch.zeros(2*ninp, ninp))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.relu = nn.ReLU()
        self.max_n= max_n
        self.device = device

    def forward(self, x, adj):
        """
        x is the postiion list, size = (batch, max_n, max_n): 
        one-hot of position, and nodes after the end type are all zeros embedding
        adj is the adjacency matrix (not the sparse matrix)
  
        #bsize = len(x)
        pos_one_hot = torch.zeros(bsize,self.max_n,self.max_n).to(self._get_device())
        for i in range(bsize):
        pos_one_hot[i,:len(x[i]),:] = self._one_hot(x[i],self.max_n)
        """
        pos_embed = torch.cat((x, torch.matmul(adj.transpose(1,2),x)),2) # concat(x_i, sum_j{x_j, j \in N(i)})
        pos_embed = self.relu(torch.matmul(pos_embed,self.W1.to(self._get_device())))
        if self.dropout > 0.0001:
            pos_embed = self.droplayer(pos_embed)
        pos_embed = torch.matmul(pos_embed,self.W2.to(self._get_device()))
        if self.dropout > 0.0001:
            pos_embed = self.droplayer(pos_embed)
        return pos_embed

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
class PACE_VAE_nomask(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, START_SYMBOL, ninp=256, nhead=8, nhid=512, nlayers=6, dropout=0.25, fc_hidden=256, nz = 64):
        super(PACE_VAE_nomask,self).__init__()
        self.max_n = max_n # maximum number of vertices (each node, node type sequence must be 2, 0,.....,1. then we could use all zeros to pad)
        self.nvt = nvt  # number of vertex types (nvt including the start node type (0), the end node (1), the start sign(2))
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.START_SYMBOL = START_SYMBOL
        self.ninp =  ninp # size of node type embedding (so as the position embedding)
        self.nhead = nhead # number of heads in multi-head atention
        self.nhid = nhid # feedforward network hidden state size (assert nhid = 2 * ninp)
        self.nz = nz # latent space dimension
        self.nlayers = nlayers # number pf transformer layers
        if dropout > 0.0001:
            self.droplayer = nn.Dropout(p=dropout)
        self.dropout = dropout 
        self.device = None

        # 1. encoder-related  
        self.pos_embed = GNNposEncoding(ninp,dropout,max_n,self.device)
        self.node_embed = nn.Sequential(
            nn.Linear(nvt, ninp),      
            nn.ReLU()
            )
        encoder_layers = TransformerEncoder_layer(nhid, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.att_agg = Attention_out_computation(nhid, nhead, nhid * 4)
        
        hidden_size = self.nhid*self.max_n
        self.hidden_size = hidden_size
        #self.fc1 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        #self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        #
        self.fc1 = nn.Linear(hidden_size,nz)
        self.fc2 = nn.Linear(hidden_size,nz)
        #self.fc1 = nn.Linear(self.nhid ,nz)
        #self.fc2 = nn.Linear(self.nhid ,nz)
        # nn.Linear(self.nhid ,nz) = nn.Linear( 4 *self.nhid ,nz)
        # nn.Linear(self.nhid ,nz) = nn.Linear(4 * self.nhid ,nz)
        #nn.Linear(hidden_size,nz) 
        #self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz))  # latent logvar

        # 2. decoder-related
        decoder_layers = TransformerDecoder_layer(nhid, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers,nlayers)

        self.add_node = nn.Sequential(
            nn.Linear(nhid,fc_hidden,bias=False),
            nn.ReLU(),
            nn.Linear(fc_hidden, nvt,bias=False)
            )
        self.add_edge = nn.Sequential(
                nn.Linear(nhid * 2, nhid), 
                nn.ReLU(), 
                nn.Linear(nhid, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew

        #self.fc3 = nn.Sequential(
        #    nn.Linear(nz,2*nz,bias=False),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,hidden_size,bias=False))
        self.fc3 = nn.Linear(nz, hidden_size)
        # 4. others
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax2 = nn.LogSoftmax(2)
        self.logsoftmax1 = nn.LogSoftmax(1)

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self._get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self._get_device())
        return x

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _mask_generate(self,adj, num_node):
        """
        compute the tgt_mask for the decoder. (already been put on the GPU)
        adj type: FloatTensor of the adjacency matrix
        """
        mask = torch.zeros_like(adj).to(self._get_device())
        mem = torch.zeros_like(adj).to(self._get_device())
        ite = 1
        mask += adj
        mem += adj
        while ite <= num_node-2 and mem.to(torch.uint8).any():
            mem = torch.matmul(mem,adj)
            mask += mem
            #print(ite)
            ite += 1
        del mem
        mask += torch.diag(torch.ones(num_node)).to(self._get_device())
        #mask = mask < 0.5
        #mask = mask.to(torch.bool).t()
        mask = mask < 0.5
        return mask

    def _get_edge_score(self, H):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(H))

    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self._get_device()) # get a zero hidden state


    def _prepare_features(self,glist,mem_len=None):
        """
        prepare the input node features, adjacency matrix, masks.
        """
        bsize = len(glist)
        node_feature = torch.zeros(bsize,self.max_n,self.nvt).to(self._get_device()) # we take one-hot encoding as the initial features
        pos_one_hot = torch.zeros(bsize,self.max_n,self.max_n).to(self._get_device()) # position encoding
        adj = torch.zeros(bsize,self.max_n,self.max_n).to(self._get_device()) # adjacency matrix
        src_mask = torch.ones(bsize * self.nhead,self.max_n-1,self.max_n-1).to(self._get_device()) # source mask
        #src_mask = torch.zeros(bsize * self.nhead,self.max_n,self.max_n).to(self._get_device()) # source mask
        tgt_mask = torch.ones(bsize * self.nhead,self.max_n,self.max_n).to(self._get_device()) # target mask
        mem_mask = torch.ones(bsize * self.nhead,self.max_n,self.max_n-1).to(self._get_device()) # target mask
        graph_sizes = [] # number of node in each graph
        true_types = [] # true graph types
        
        head_count = 0
        for i in range(bsize):
            g = glist[i]
            ntype = g.vs['type']
            ptype = g.vs['type']
            num_node = len(ntype)
            if num_node < self.max_n:
                ntype += [self.END_TYPE] * (self.max_n - num_node)
                ptype += [self.END_TYPE] * (self.max_n - num_node)
            ############
            #dict_i = dict(zip(ptype,np.arange(len(ptype))))
            #sort_dici = sorted(dict_i.items(),key=lambda item:item[0])
            #idx_i = [j for i,j in sort_dici]
            ##############
            # node i feature
            ntype_one_hot = self._one_hot(ntype,self.nvt)
            #node_feature[i,:,:] = ntype_one_hot[torch.LongTensor(idx_i),:] 
            node_feature[i,:,:] = ntype_one_hot # the 'extra' nodes are padded with the zero embeddings
            # position one-hot
            #pos_one_hot[i,:,:] = self._one_hot(ptype,self.max_n)[torch.LongTensor(idx_i),:]
            pos_one_hot[i,:,:] = self._one_hot(ptype,self.max_n)
            # node i adj
            #adj_i = torch.FloatTensor(g.get_adjacency().data)[torch.LongTensor(idx_i),:][:,torch.LongTensor(idx_i)].to(self._get_device())
            adj_i = torch.FloatTensor(g.get_adjacency().data).to(self._get_device())
            adj[i,:num_node,:num_node] = adj_i
            # src mask
            src_mask[head_count:head_count+self.nhead,:num_node-1,:num_node-1] = torch.zeros(self.nhead,num_node-1,num_node-1).to(self._get_device()) # no such start symbol node
            src_mask[head_count:head_count+self.nhead,num_node-1:,num_node-1:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-num_node).to(self._get_device()) # no such start symbol node            
            # tgt mask
            tgt_mask[head_count:head_count+self.nhead,:num_node,:num_node] = torch.stack([self._mask_generate(adj_i,num_node)]*self.nhead,0)
            tgt_mask[head_count:head_count+self.nhead,num_node:,num_node:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-num_node).to(self._get_device())
            # memory mask
            if mem_len == None:
                mem_len = num_node-1
                mem_mask[head_count:head_count+self.nhead,:num_node,:mem_len] = torch.zeros(self.nhead,num_node,mem_len).to(self._get_device())
                mem_mask[head_count:head_count+self.nhead,num_node:,mem_len:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-1-mem_len).to(self._get_device())
            else:
                mem_mask[head_count:head_count+self.nhead,:num_node,:mem_len] = torch.zeros(self.nhead,num_node,mem_len).to(self._get_device())
                mem_mask[head_count:head_count+self.nhead,num_node:,-1:] = torch.zeros(self.nhead,self.max_n-num_node,1).to(self._get_device())
            # graph size
            graph_sizes.append(g.vcount()) # graph size = number of node + 2 (start type and a end type )
            # true type
            true_types.append(g.vs['type'][1:]) # we skip the start node for teacher forcing
            #true_types.append(list(np.array(g.vs['type'])[idx_i])[1:]) 
            head_count += self.nhead
        return node_feature, pos_one_hot, adj, src_mask.to(torch.bool), tgt_mask.to(torch.bool).transpose(1,2), mem_mask.to(torch.bool), graph_sizes, true_types

    def encode(self, glist):
        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        bsize = len(graph_sizes) 

        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        # here we set the source sequence and the tgt sequence for the teacher forcing
        #src_inp = node_feat[:,1:,:].transpose(0,1) # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)
        src_inp = node_feat.transpose(0,1)

        #memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp,mask=tgt_mask).transpose(0,1)
        
        out = torch.zeros_like(memory).to(self._get_device())
        for k in range(out.size(0)): # cannonical order
            type_m1 = glist[k].vs['type']
            dict_i = dict(zip(type_m1,np.arange(len(type_m1))))
            sort_dici = sorted(dict_i.items(),key=lambda item:item[0])
            idx_i = [j for i,j in sort_dici]
            out[k,:,:] = memory[k,torch.LongTensor(idx_i),:]
        out = out.reshape(-1,self.max_n*self.nhid) #shape ( bsize, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.
        # memory = memory.transpose(0,1).reshape(-1,(self.max_n-1)*self.nhid)
        return self.fc1(out), self.fc2(out)

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self,z):
        """
        This is a sequence to sequence prediction model.
        Input: a graph (sequence of nodes)
        from a START_TYPE node, we use the transformer to predict the type of the next node
        and this process is continued until the END_TYPE node (or iterations reaches max_n)
        """
        bsize = len(z)
        out_de = self.fc3(z).reshape(-1,self.max_n,self.nhid)
        #memory = self.fc3(z).reshape(-1,self.max_n-1,self.nhid).transpose(0,1)

        G = [igraph.Graph(directed=True) for _ in range(bsize)]
        for g in G:
            g.add_vertex(type=self.START_SYMBOL)
            g.add_vertex(type=self.START_TYPE)
            g.add_edge(0,1)

        #memory = self.encoder(src_inp,mask=src_mask)

        finished = [False] * bsize
        for idx in range(2, self.max_n): # the first two type of nodes are certain
            node_one_hot, pos_one_hot, adj, _, tgt_mask, mem_mask, _, _ = self._prepare_features(G,self.max_n-1)
            memory = torch.zeros_like(out_de).to(self._get_device())
            for k in range(bsize):
                Gk_type = G[k].vs['type']
                if len(Gk_type) < self.max_n:
                    Gk_type += [self.END_TYPE] * (self.max_n-len(Gk_type))
                memory[k,:,:] = out_de[k,torch.LongTensor(Gk_type),:]
            memory = memory.transpose(0,1)
            
            pos_feat = self.pos_embed(pos_one_hot, adj)
            node_feat = self.node_embed(node_one_hot)
            node_feat = torch.cat([node_feat,pos_feat],2)
            tgt_inp = node_feat.transpose(0,1)

            out = self.decoder(tgt_inp,memory,tgt_mask=tgt_mask,memory_mask=mem_mask)
            out = out.transpose(0,1) #shape ( bsize, self.max_n, nvrt)
            next_node_hidden = out[:,idx-1,:]
            # add nodes
            type_scores = self.add_node(next_node_hidden)
            type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
            new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) for i in range(len(G))]
            # add edges 
            edge_scores = torch.cat([torch.stack([next_node_hidden]*(idx-1),1), out[:,:idx-1,:]],-1) # just from the cneter node to the target node
            edge_scores = self._get_edge_score(edge_scores)
            
            for i, g in enumerate(G):
                if not finished[i]:
                    if idx < self.max_n-1:
                        g.add_vertex(type=new_types[i])
                    else:
                        g.add_vertex(type=self.END_TYPE)
            for vi in range(idx-2, -1, -1):
                ei_score = edge_scores[:,vi] # 0 point to node 1
                random_score = torch.rand_like(ei_score)
                decisions = random_score < ei_score
                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE: 
                        # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                            if v.index != g.vcount()-1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount()-1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi+1, g.vcount()-1)
            for i, g in enumerate(G):
                _ = longest_path(g)
        return G

    def loss(self, mu, logvar, glist, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar) # (bsize, hidden)
        #memory = self.fc3(z).reshape(-1,self.max_n-1,self.nhid).transpose(0,1)

        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        out = self.fc3(z).reshape(-1,self.max_n,self.nhid)
        memory = torch.zeros_like(out).to(self._get_device())
        for k in range(out.size(0)):
            memory[k,:,:] = out[k,torch.LongTensor(glist[k].vs['type'])]
        memory = memory.transpose(0,1)
        
        bsize = len(graph_sizes)
        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        tgt_inp = node_feat.transpose(0,1)
        out = self.decoder(tgt_inp,memory,tgt_mask=tgt_mask,memory_mask=mem_mask) # shape (self.max_n, bsize, nhid)
        out = out.transpose(0,1)

        scores = self.add_node(out)
        scores = self.logsoftmax2(scores)  # shape ( bsize, self.max_n, nvrt)
        res = 0 # loglikelihood
        for i in range(bsize):
            # vertex log likelihood
            #print(true_types[i])
            if len(true_types[i]) < self.max_n:
                true_types[i] += [0] * (self.max_n-len(true_types[i]))
            vll = scores[i][np.arange(self.max_n),true_types[i]][:graph_sizes[i]-1].sum() # only count 'no padding' nodes. graph size i - 1 since the input symbol of the encoder do not have the start node
            res += vll
            # edges log likelihood
            num_node_i = graph_sizes[i]-1 # no start node
            num_pot_edges = int(num_node_i*(num_node_i-1)/2.0)
            edge_scores = torch.zeros(num_pot_edges,2*self.nhid).to(self._get_device())
            ground_truth = torch.zeros(num_pot_edges,1).to(self._get_device())
            count = 0
            for idx in range(num_node_i-1,0,-1):
                edge_scores[count:count + idx, :] = torch.cat([torch.stack([out[i,idx,:]]*idx,0), out[i,:idx,:]],-1) # in each batch, ith row of out represent the presentation of node i+1 (since input do not have the start node)
                ground_truth[count:count + idx, :] = adj[i,1:idx+1,idx+1].view(idx,1)
                count += idx

            edge_scores = self._get_edge_score(edge_scores)
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum') 
            res += ell

        res = -res  # convert likelihood to loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss + res
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.hidden_size).to(self._get_device())
        G = self.decode(sample)
        return G
    
class PACE_VAE_nodagseq(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, START_SYMBOL, ninp=256, nhead=8, nhid=512, nlayers=6, dropout=0.25, fc_hidden=256, nz = 64):
        super(PACE_VAE_nodagseq,self).__init__()
        self.max_n = max_n # maximum number of vertices (each node, node type sequence must be 2, 0,.....,1. then we could use all zeros to pad)
        self.nvt = nvt  # number of vertex types (nvt including the start node type (0), the end node (1), the start sign(2))
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.START_SYMBOL = START_SYMBOL
        self.ninp =  ninp # size of node type embedding (so as the position embedding)
        self.nhead = nhead # number of heads in multi-head atention
        self.nhid = nhid # feedforward network hidden state size (assert nhid = 2 * ninp)
        self.nz = nz # latent space dimension
        self.nlayers = nlayers # number pf transformer layers
        if dropout > 0.0001:
            self.droplayer = nn.Dropout(p=dropout)
        self.dropout = dropout 
        self.device = None

        # 1. encoder-related  
        self.pos_embed = nn.Linear(max_n,ninp)
        self.node_embed = nn.Sequential(
            nn.Linear(nvt, ninp),      
            nn.ReLU()
            )
        encoder_layers = TransformerEncoder_layer(nhid, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.att_agg = Attention_out_computation(nhid, nhead, nhid * 4)
        
        hidden_size = self.nhid*self.max_n
        self.hidden_size = hidden_size
        #self.fc1 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        #self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        #
        self.fc1 = nn.Linear(hidden_size,nz)
        self.fc2 = nn.Linear(hidden_size,nz)
        #self.fc1 = nn.Linear(self.nhid ,nz)
        #self.fc2 = nn.Linear(self.nhid ,nz)
        # nn.Linear(self.nhid ,nz) = nn.Linear( 4 *self.nhid ,nz)
        # nn.Linear(self.nhid ,nz) = nn.Linear(4 * self.nhid ,nz)
        #nn.Linear(hidden_size,nz) 
        #self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz))  # latent logvar

        # 2. decoder-related
        decoder_layers = TransformerDecoder_layer(nhid, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers,nlayers)

        self.add_node = nn.Sequential(
            nn.Linear(nhid,fc_hidden,bias=False),
            nn.ReLU(),
            nn.Linear(fc_hidden, nvt,bias=False)
            )
        self.add_edge = nn.Sequential(
                nn.Linear(nhid * 2, nhid), 
                nn.ReLU(), 
                nn.Linear(nhid, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew

        #self.fc3 = nn.Sequential(
        #    nn.Linear(nz,2*nz,bias=False),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,hidden_size,bias=False))
        self.fc3 = nn.Linear(nz, hidden_size)
        # 4. others
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax2 = nn.LogSoftmax(2)
        self.logsoftmax1 = nn.LogSoftmax(1)

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self._get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self._get_device())
        return x

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _mask_generate(self,adj, num_node):
        """
        compute the tgt_mask for the decoder. (already been put on the GPU)
        adj type: FloatTensor of the adjacency matrix
        """
        mask = torch.zeros_like(adj).to(self._get_device())
        mem = torch.zeros_like(adj).to(self._get_device())
        ite = 1
        mask += adj
        mem += adj
        while ite <= num_node-2 and mem.to(torch.uint8).any():
            mem = torch.matmul(mem,adj)
            mask += mem
            #print(ite)
            ite += 1
        del mem
        mask += torch.diag(torch.ones(num_node)).to(self._get_device())
        #mask = mask < 0.5
        #mask = mask.to(torch.bool).t()
        mask = mask < 0.5
        return mask

    def _get_edge_score(self, H):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(H))

    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self._get_device()) # get a zero hidden state


    def _prepare_features(self,glist,mem_len=None):
        """
        prepare the input node features, adjacency matrix, masks.
        """
        bsize = len(glist)
        node_feature = torch.zeros(bsize,self.max_n,self.nvt).to(self._get_device()) # we take one-hot encoding as the initial features
        pos_one_hot = torch.zeros(bsize,self.max_n,self.max_n).to(self._get_device()) # position encoding
        adj = torch.zeros(bsize,self.max_n,self.max_n).to(self._get_device()) # adjacency matrix
        src_mask = torch.ones(bsize * self.nhead,self.max_n-1,self.max_n-1).to(self._get_device()) # source mask
        #src_mask = torch.zeros(bsize * self.nhead,self.max_n,self.max_n).to(self._get_device()) # source mask
        tgt_mask = torch.ones(bsize * self.nhead,self.max_n,self.max_n).to(self._get_device()) # target mask
        mem_mask = torch.ones(bsize * self.nhead,self.max_n,self.max_n-1).to(self._get_device()) # target mask
        graph_sizes = [] # number of node in each graph
        true_types = [] # true graph types
        
        head_count = 0
        for i in range(bsize):
            g = glist[i]
            ntype = g.vs['type']
            ptype = g.vs['type']
            num_node = len(ntype)
            if num_node < self.max_n:
                ntype += [self.END_TYPE] * (self.max_n - num_node)
                ptype += [self.END_TYPE] * (self.max_n - num_node)
            ############
            #dict_i = dict(zip(ptype,np.arange(len(ptype))))
            #sort_dici = sorted(dict_i.items(),key=lambda item:item[0])
            #idx_i = [j for i,j in sort_dici]
            ##############
            # node i feature
            ntype_one_hot = self._one_hot(ntype,self.nvt)
            #node_feature[i,:,:] = ntype_one_hot[torch.LongTensor(idx_i),:] 
            node_feature[i,:,:] = ntype_one_hot # the 'extra' nodes are padded with the zero embeddings
            # position one-hot
            #pos_one_hot[i,:,:] = self._one_hot(ptype,self.max_n)[torch.LongTensor(idx_i),:]
            pos_one_hot[i,:,:] = self._one_hot(ptype,self.max_n)
            # node i adj
            #adj_i = torch.FloatTensor(g.get_adjacency().data)[torch.LongTensor(idx_i),:][:,torch.LongTensor(idx_i)].to(self._get_device())
            adj_i = torch.FloatTensor(g.get_adjacency().data).to(self._get_device())
            adj[i,:num_node,:num_node] = adj_i
            # src mask
            src_mask[head_count:head_count+self.nhead,:num_node-1,:num_node-1] = torch.zeros(self.nhead,num_node-1,num_node-1).to(self._get_device()) # no such start symbol node
            src_mask[head_count:head_count+self.nhead,num_node-1:,num_node-1:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-num_node).to(self._get_device()) # no such start symbol node            
            # tgt mask
            tgt_mask[head_count:head_count+self.nhead,:num_node,:num_node] = torch.stack([self._mask_generate(adj_i,num_node)]*self.nhead,0)
            tgt_mask[head_count:head_count+self.nhead,num_node:,num_node:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-num_node).to(self._get_device())
            # memory mask
            if mem_len == None:
                mem_len = num_node-1
                mem_mask[head_count:head_count+self.nhead,:num_node,:mem_len] = torch.zeros(self.nhead,num_node,mem_len).to(self._get_device())
                mem_mask[head_count:head_count+self.nhead,num_node:,mem_len:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-1-mem_len).to(self._get_device())
            else:
                mem_mask[head_count:head_count+self.nhead,:num_node,:mem_len] = torch.zeros(self.nhead,num_node,mem_len).to(self._get_device())
                mem_mask[head_count:head_count+self.nhead,num_node:,-1:] = torch.zeros(self.nhead,self.max_n-num_node,1).to(self._get_device())
            # graph size
            graph_sizes.append(g.vcount()) # graph size = number of node + 2 (start type and a end type )
            # true type
            true_types.append(g.vs['type'][1:]) # we skip the start node for teacher forcing
            #true_types.append(list(np.array(g.vs['type'])[idx_i])[1:]) 
            head_count += self.nhead
        return node_feature, pos_one_hot, adj, src_mask.to(torch.bool), tgt_mask.to(torch.bool).transpose(1,2), mem_mask.to(torch.bool), graph_sizes, true_types

    def encode(self, glist):
        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        bsize = len(graph_sizes) 

        pos_feat = self.pos_embed(pos_one_hot)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        # here we set the source sequence and the tgt sequence for the teacher forcing
        #src_inp = node_feat[:,1:,:].transpose(0,1) # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)
        src_inp = node_feat.transpose(0,1)

        #memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp,mask=tgt_mask).transpose(0,1)
        
        out = torch.zeros_like(memory).to(self._get_device())
        for k in range(out.size(0)): # cannonical order
            type_m1 = glist[k].vs['type']
            dict_i = dict(zip(type_m1,np.arange(len(type_m1))))
            sort_dici = sorted(dict_i.items(),key=lambda item:item[0])
            idx_i = [j for i,j in sort_dici]
            out[k,:,:] = memory[k,torch.LongTensor(idx_i),:]
        out = out.reshape(-1,self.max_n*self.nhid) #shape ( bsize, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.
        # memory = memory.transpose(0,1).reshape(-1,(self.max_n-1)*self.nhid)
        return self.fc1(out), self.fc2(out)

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self,z):
        """
        This is a sequence to sequence prediction model.
        Input: a graph (sequence of nodes)
        from a START_TYPE node, we use the transformer to predict the type of the next node
        and this process is continued until the END_TYPE node (or iterations reaches max_n)
        """
        bsize = len(z)
        out_de = self.fc3(z).reshape(-1,self.max_n,self.nhid)
        #memory = self.fc3(z).reshape(-1,self.max_n-1,self.nhid).transpose(0,1)

        G = [igraph.Graph(directed=True) for _ in range(bsize)]
        for g in G:
            g.add_vertex(type=self.START_SYMBOL)
            g.add_vertex(type=self.START_TYPE)
            g.add_edge(0,1)

        #memory = self.encoder(src_inp,mask=src_mask)

        finished = [False] * bsize
        for idx in range(2, self.max_n): # the first two type of nodes are certain
            node_one_hot, pos_one_hot, adj, _, tgt_mask, mem_mask, _, _ = self._prepare_features(G,self.max_n-1)
            memory = torch.zeros_like(out_de).to(self._get_device())
            for k in range(bsize):
                Gk_type = G[k].vs['type']
                if len(Gk_type) < self.max_n:
                    Gk_type += [self.END_TYPE] * (self.max_n-len(Gk_type))
                memory[k,:,:] = out_de[k,torch.LongTensor(Gk_type),:]
            memory = memory.transpose(0,1)
            
            pos_feat = self.pos_embed(pos_one_hot)
            node_feat = self.node_embed(node_one_hot)
            node_feat = torch.cat([node_feat,pos_feat],2)
            tgt_inp = node_feat.transpose(0,1)

            out = self.decoder(tgt_inp,memory,tgt_mask=tgt_mask,memory_mask=mem_mask)
            out = out.transpose(0,1) #shape ( bsize, self.max_n, nvrt)
            next_node_hidden = out[:,idx-1,:]
            # add nodes
            type_scores = self.add_node(next_node_hidden)
            type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
            new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) for i in range(len(G))]
            # add edges 
            edge_scores = torch.cat([torch.stack([next_node_hidden]*(idx-1),1), out[:,:idx-1,:]],-1) # just from the cneter node to the target node
            edge_scores = self._get_edge_score(edge_scores)
            
            for i, g in enumerate(G):
                if not finished[i]:
                    if idx < self.max_n-1:
                        g.add_vertex(type=new_types[i])
                    else:
                        g.add_vertex(type=self.END_TYPE)
            for vi in range(idx-2, -1, -1):
                ei_score = edge_scores[:,vi] # 0 point to node 1
                random_score = torch.rand_like(ei_score)
                decisions = random_score < ei_score
                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE: 
                        # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                            if v.index != g.vcount()-1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount()-1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi+1, g.vcount()-1)
            for i, g in enumerate(G):
                _ = longest_path(g)
        return G

    def loss(self, mu, logvar, glist, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar) # (bsize, hidden)
        #memory = self.fc3(z).reshape(-1,self.max_n-1,self.nhid).transpose(0,1)

        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        out = self.fc3(z).reshape(-1,self.max_n,self.nhid)
        memory = torch.zeros_like(out).to(self._get_device())
        for k in range(out.size(0)):
            memory[k,:,:] = out[k,torch.LongTensor(glist[k].vs['type'])]
        memory = memory.transpose(0,1)
        
        bsize = len(graph_sizes)
        pos_feat = self.pos_embed(pos_one_hot)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        tgt_inp = node_feat.transpose(0,1)
        out = self.decoder(tgt_inp,memory,tgt_mask=tgt_mask,memory_mask=mem_mask) # shape (self.max_n, bsize, nhid)
        out = out.transpose(0,1)

        scores = self.add_node(out)
        scores = self.logsoftmax2(scores)  # shape ( bsize, self.max_n, nvrt)
        res = 0 # loglikelihood
        for i in range(bsize):
            # vertex log likelihood
            #print(true_types[i])
            if len(true_types[i]) < self.max_n:
                true_types[i] += [0] * (self.max_n-len(true_types[i]))
            vll = scores[i][np.arange(self.max_n),true_types[i]][:graph_sizes[i]-1].sum() # only count 'no padding' nodes. graph size i - 1 since the input symbol of the encoder do not have the start node
            res += vll
            # edges log likelihood
            num_node_i = graph_sizes[i]-1 # no start node
            num_pot_edges = int(num_node_i*(num_node_i-1)/2.0)
            edge_scores = torch.zeros(num_pot_edges,2*self.nhid).to(self._get_device())
            ground_truth = torch.zeros(num_pot_edges,1).to(self._get_device())
            count = 0
            for idx in range(num_node_i-1,0,-1):
                edge_scores[count:count + idx, :] = torch.cat([torch.stack([out[i,idx,:]]*idx,0), out[i,:idx,:]],-1) # in each batch, ith row of out represent the presentation of node i+1 (since input do not have the start node)
                ground_truth[count:count + idx, :] = adj[i,1:idx+1,idx+1].view(idx,1)
                count += idx

            edge_scores = self._get_edge_score(edge_scores)
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum') 
            res += ell

        res = -res  # convert likelihood to loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss + res
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.hidden_size).to(self._get_device())
        G = self.decode(sample)
        return G
    
class PACE_VAE(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, START_SYMBOL, ninp=256, nhead=8, nhid=512, nlayers=6, dropout=0.25, fc_hidden=256, nz = 64):
        super(PACE_VAE,self).__init__()
        self.max_n = max_n # maximum number of vertices (each node, node type sequence must be 2, 0,.....,1. then we could use all zeros to pad)
        self.nvt = nvt  # number of vertex types (nvt including the start node type (0), the end node (1), the start sign(2))
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.START_SYMBOL = START_SYMBOL
        self.ninp =  ninp # size of node type embedding (so as the position embedding)
        self.nhead = nhead # number of heads in multi-head atention
        self.nhid = nhid # feedforward network hidden state size (assert nhid = 2 * ninp)
        self.nz = nz # latent space dimension
        self.nlayers = nlayers # number pf transformer layers
        if dropout > 0.0001:
            self.droplayer = nn.Dropout(p=dropout)
        self.dropout = dropout 
        self.device = None

        # 1. encoder-related  
        self.pos_embed = GNNposEncoding(ninp,dropout,max_n,self.device)
        self.node_embed = nn.Sequential(
            nn.Linear(nvt, ninp),      
            nn.ReLU()
            )
        encoder_layers = TransformerEncoder_layer(nhid, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.att_agg = Attention_out_computation(nhid, nhead, nhid * 4)
        
        hidden_size = self.nhid*self.max_n
        self.hidden_size = hidden_size
        #self.fc1 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        #self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        #
        self.fc1 = nn.Linear(hidden_size,nz)
        self.fc2 = nn.Linear(hidden_size,nz)
        #self.fc1 = nn.Linear(self.nhid ,nz)
        #self.fc2 = nn.Linear(self.nhid ,nz)
        # nn.Linear(self.nhid ,nz) = nn.Linear( 4 *self.nhid ,nz)
        # nn.Linear(self.nhid ,nz) = nn.Linear(4 * self.nhid ,nz)
        #nn.Linear(hidden_size,nz) 
        #self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz))  # latent logvar

        # 2. decoder-related
        decoder_layers = TransformerDecoder_layer(nhid, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers,nlayers)

        self.add_node = nn.Sequential(
            nn.Linear(nhid,fc_hidden,bias=False),
            nn.ReLU(),
            nn.Linear(fc_hidden, nvt,bias=False)
            )
        self.add_edge = nn.Sequential(
                nn.Linear(nhid * 2, nhid), 
                nn.ReLU(), 
                nn.Linear(nhid, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew

        #self.fc3 = nn.Sequential(
        #    nn.Linear(nz,2*nz,bias=False),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,hidden_size,bias=False))
        self.fc3 = nn.Linear(nz, hidden_size)
        # 4. others
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax2 = nn.LogSoftmax(2)
        self.logsoftmax1 = nn.LogSoftmax(1)

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self._get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self._get_device())
        return x

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _mask_generate(self,adj, num_node):
        """
        compute the tgt_mask for the decoder. (already been put on the GPU)
        adj type: FloatTensor of the adjacency matrix
        """
        mask = torch.zeros_like(adj).to(self._get_device())
        mem = torch.zeros_like(adj).to(self._get_device())
        ite = 1
        mask += adj
        mem += adj
        while ite <= num_node-2 and mem.to(torch.uint8).any():
            mem = torch.matmul(mem,adj)
            mask += mem
            #print(ite)
            ite += 1
        del mem
        mask += torch.diag(torch.ones(num_node)).to(self._get_device())
        #mask = mask < 0.5
        #mask = mask.to(torch.bool).t()
        mask = mask < 0.5
        return mask

    def _get_edge_score(self, H):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(H))

    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self._get_device()) # get a zero hidden state


    def _prepare_features(self,glist,mem_len=None):
        """
        prepare the input node features, adjacency matrix, masks.
        """
        bsize = len(glist)
        node_feature = torch.zeros(bsize,self.max_n,self.nvt).to(self._get_device()) # we take one-hot encoding as the initial features
        pos_one_hot = torch.zeros(bsize,self.max_n,self.max_n).to(self._get_device()) # position encoding
        adj = torch.zeros(bsize,self.max_n,self.max_n).to(self._get_device()) # adjacency matrix
        src_mask = torch.ones(bsize * self.nhead,self.max_n-1,self.max_n-1).to(self._get_device()) # source mask
        #src_mask = torch.zeros(bsize * self.nhead,self.max_n,self.max_n).to(self._get_device()) # source mask
        tgt_mask = torch.ones(bsize * self.nhead,self.max_n,self.max_n).to(self._get_device()) # target mask
        mem_mask = torch.ones(bsize * self.nhead,self.max_n,self.max_n-1).to(self._get_device()) # target mask
        graph_sizes = [] # number of node in each graph
        true_types = [] # true graph types
        
        head_count = 0
        for i in range(bsize):
            g = glist[i]
            ntype = g.vs['type']
            ptype = g.vs['type']
            num_node = len(ntype)
            if num_node < self.max_n:
                ntype += [self.END_TYPE] * (self.max_n - num_node)
                ptype += [self.END_TYPE] * (self.max_n - num_node)
            ############
            #dict_i = dict(zip(ptype,np.arange(len(ptype))))
            #sort_dici = sorted(dict_i.items(),key=lambda item:item[0])
            #idx_i = [j for i,j in sort_dici]
            ##############
            # node i feature
            ntype_one_hot = self._one_hot(ntype,self.nvt)
            #node_feature[i,:,:] = ntype_one_hot[torch.LongTensor(idx_i),:] 
            node_feature[i,:,:] = ntype_one_hot # the 'extra' nodes are padded with the zero embeddings
            # position one-hot
            #pos_one_hot[i,:,:] = self._one_hot(ptype,self.max_n)[torch.LongTensor(idx_i),:]
            pos_one_hot[i,:,:] = self._one_hot(ptype,self.max_n)
            # node i adj
            #adj_i = torch.FloatTensor(g.get_adjacency().data)[torch.LongTensor(idx_i),:][:,torch.LongTensor(idx_i)].to(self._get_device())
            adj_i = torch.FloatTensor(g.get_adjacency().data).to(self._get_device())
            adj[i,:num_node,:num_node] = adj_i
            # src mask
            src_mask[head_count:head_count+self.nhead,:num_node-1,:num_node-1] = torch.stack([self._mask_generate(adj_i,num_node)[1:,1:]]*self.nhead,0) 
            src_mask[head_count:head_count+self.nhead,num_node-1:,num_node-1:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-num_node).to(self._get_device())
            # tgt mask
            tgt_mask[head_count:head_count+self.nhead,:num_node,:num_node] = torch.stack([self._mask_generate(adj_i,num_node)]*self.nhead,0)
            tgt_mask[head_count:head_count+self.nhead,num_node:,num_node:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-num_node).to(self._get_device())
            # memory mask
            if mem_len == None:
                mem_len = num_node-1
                mem_mask[head_count:head_count+self.nhead,:num_node,:mem_len] = torch.zeros(self.nhead,num_node,mem_len).to(self._get_device())
                mem_mask[head_count:head_count+self.nhead,num_node:,mem_len:] = torch.zeros(self.nhead,self.max_n-num_node,self.max_n-1-mem_len).to(self._get_device())
            else:
                mem_mask[head_count:head_count+self.nhead,:num_node,:mem_len] = torch.zeros(self.nhead,num_node,mem_len).to(self._get_device())
                mem_mask[head_count:head_count+self.nhead,num_node:,-1:] = torch.zeros(self.nhead,self.max_n-num_node,1).to(self._get_device())
            # graph size
            graph_sizes.append(g.vcount()) # graph size = number of node + 2 (start type and a end type )
            # true type
            true_types.append(g.vs['type'][1:]) # we skip the start node for teacher forcing
            #true_types.append(list(np.array(g.vs['type'])[idx_i])[1:]) 
            head_count += self.nhead
        return node_feature, pos_one_hot, adj, src_mask.to(torch.bool), tgt_mask.to(torch.bool).transpose(1,2), mem_mask.to(torch.bool), graph_sizes, true_types

    def encode(self, glist):
        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        bsize = len(graph_sizes) 

        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        # here we set the source sequence and the tgt sequence for the teacher forcing
        #src_inp = node_feat[:,1:,:].transpose(0,1) # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)
        src_inp = node_feat.transpose(0,1)

        #memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp,mask=tgt_mask).transpose(0,1)
        
        out = torch.zeros_like(memory).to(self._get_device())
        for k in range(out.size(0)): # cannonical order
            type_m1 = glist[k].vs['type']
            dict_i = dict(zip(type_m1,np.arange(len(type_m1))))
            sort_dici = sorted(dict_i.items(),key=lambda item:item[0])
            idx_i = [j for i,j in sort_dici]
            out[k,:,:] = memory[k,torch.LongTensor(idx_i),:]
        out = out.reshape(-1,self.max_n*self.nhid) #shape ( bsize, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.
        # memory = memory.transpose(0,1).reshape(-1,(self.max_n-1)*self.nhid)
        return self.fc1(out), self.fc2(out)

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self,z):
        """
        This is a sequence to sequence prediction model.
        Input: a graph (sequence of nodes)
        from a START_TYPE node, we use the transformer to predict the type of the next node
        and this process is continued until the END_TYPE node (or iterations reaches max_n)
        """
        bsize = len(z)
        out_de = self.fc3(z).reshape(-1,self.max_n,self.nhid)
        #memory = self.fc3(z).reshape(-1,self.max_n-1,self.nhid).transpose(0,1)

        G = [igraph.Graph(directed=True) for _ in range(bsize)]
        for g in G:
            g.add_vertex(type=self.START_SYMBOL)
            g.add_vertex(type=self.START_TYPE)
            g.add_edge(0,1)

        #memory = self.encoder(src_inp,mask=src_mask)

        finished = [False] * bsize
        for idx in range(2, self.max_n): # the first two type of nodes are certain
            node_one_hot, pos_one_hot, adj, _, tgt_mask, mem_mask, _, _ = self._prepare_features(G,self.max_n-1)
            memory = torch.zeros_like(out_de).to(self._get_device())
            for k in range(bsize):
                Gk_type = G[k].vs['type']
                if len(Gk_type) < self.max_n:
                    Gk_type += [self.END_TYPE] * (self.max_n-len(Gk_type))
                memory[k,:,:] = out_de[k,torch.LongTensor(Gk_type),:]
            memory = memory.transpose(0,1)
            
            pos_feat = self.pos_embed(pos_one_hot, adj)
            node_feat = self.node_embed(node_one_hot)
            node_feat = torch.cat([node_feat,pos_feat],2)
            tgt_inp = node_feat.transpose(0,1)

            out = self.decoder(tgt_inp,memory,tgt_mask=tgt_mask,memory_mask=mem_mask)
            out = out.transpose(0,1) #shape ( bsize, self.max_n, nvrt)
            next_node_hidden = out[:,idx-1,:]
            # add nodes
            type_scores = self.add_node(next_node_hidden)
            type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
            new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) for i in range(len(G))]
            # add edges 
            edge_scores = torch.cat([torch.stack([next_node_hidden]*(idx-1),1), out[:,:idx-1,:]],-1) # just from the cneter node to the target node
            edge_scores = self._get_edge_score(edge_scores)
            
            for i, g in enumerate(G):
                if not finished[i]:
                    if idx < self.max_n-1:
                        g.add_vertex(type=new_types[i])
                    else:
                        g.add_vertex(type=self.END_TYPE)
            for vi in range(idx-2, -1, -1):
                ei_score = edge_scores[:,vi] # 0 point to node 1
                random_score = torch.rand_like(ei_score)
                decisions = random_score < ei_score
                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE: 
                        # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                            if v.index != g.vcount()-1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount()-1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi+1, g.vcount()-1)
            for i, g in enumerate(G):
                _ = longest_path(g)
        return G

    def loss(self, mu, logvar, glist, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar) # (bsize, hidden)
        #memory = self.fc3(z).reshape(-1,self.max_n-1,self.nhid).transpose(0,1)

        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        out = self.fc3(z).reshape(-1,self.max_n,self.nhid)
        memory = torch.zeros_like(out).to(self._get_device())
        for k in range(out.size(0)):
            memory[k,:,:] = out[k,torch.LongTensor(glist[k].vs['type'])]
        memory = memory.transpose(0,1)
        
        bsize = len(graph_sizes)
        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        tgt_inp = node_feat.transpose(0,1)
        out = self.decoder(tgt_inp,memory,tgt_mask=tgt_mask,memory_mask=mem_mask) # shape (self.max_n, bsize, nhid)
        out = out.transpose(0,1)

        scores = self.add_node(out)
        scores = self.logsoftmax2(scores)  # shape ( bsize, self.max_n, nvrt)
        res = 0 # loglikelihood
        for i in range(bsize):
            # vertex log likelihood
            #print(true_types[i])
            if len(true_types[i]) < self.max_n:
                true_types[i] += [0] * (self.max_n-len(true_types[i]))
            vll = scores[i][np.arange(self.max_n),true_types[i]][:graph_sizes[i]-1].sum() # only count 'no padding' nodes. graph size i - 1 since the input symbol of the encoder do not have the start node
            res += vll
            # edges log likelihood
            num_node_i = graph_sizes[i]-1 # no start node
            num_pot_edges = int(num_node_i*(num_node_i-1)/2.0)
            edge_scores = torch.zeros(num_pot_edges,2*self.nhid).to(self._get_device())
            ground_truth = torch.zeros(num_pot_edges,1).to(self._get_device())
            count = 0
            for idx in range(num_node_i-1,0,-1):
                edge_scores[count:count + idx, :] = torch.cat([torch.stack([out[i,idx,:]]*idx,0), out[i,:idx,:]],-1) # in each batch, ith row of out represent the presentation of node i+1 (since input do not have the start node)
                ground_truth[count:count + idx, :] = adj[i,1:idx+1,idx+1].view(idx,1)
                count += idx

            edge_scores = self._get_edge_score(edge_scores)
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum') 
            res += ell

        res = -res  # convert likelihood to loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss + res
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.hidden_size).to(self._get_device())
        G = self.decode(sample)
        return G