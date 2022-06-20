import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import igraph
import pdb

class DVAE(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, vid=True):
        super(DVAE, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.vid = vid
        self.device = None

        if self.vid:
            self.vs = hs + max_n  # vertex state size = hidden state + vid
        else:
            self.vs = hs

        # 0. encoding-related
        self.grue_forward = nn.GRUCell(nvt, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(nvt, hs)  # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.GRUCell(nvt, hs)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.ReLU(),
                nn.Linear(hs * 2, nvt)
                )  # which type of new vertex to add f(h0, hg)
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 2, hs * 4), 
                nn.ReLU(), 
                nn.Linear(hs * 4, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew)

        # 2. gate-related
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False), 
                )

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                    nn.Linear(hs * 2, hs), 
                    )
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.gs * 2, self.gs), 
                    )

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            gate, mapper = self.gate_forward, self.mapper_forward
        if self.vid:
            H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + 
                            [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) 
                            for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator, reverse=reverse)
        return Hv

    def _update_v(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False)
        return
    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        Hg = self._get_graph_state(G)
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _get_edge_score(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H], -1)))

    def decode(self, z, stochastic=True):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._update_v(G, 0, H0)
        finished = [False] * len(G)
        for idx in range(1, self.max_n):
            # decide the type of the next added vertex
            if idx == self.max_n - 1:  # force the last node to be end_type
                new_types = [self.END_TYPE] * len(G)
            else:
                Hg = self._get_graph_state(G, decode=True)
                type_scores = self.add_vertex(Hg)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) 
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            for i, g in enumerate(G):
                if not finished[i]:
                    g.add_vertex(type=new_types[i])
            self._update_v(G, idx)

            # decide connections
            edge_scores = []
            for vi in range(idx-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, idx)
                ei_score = self._get_edge_score(Hvi, H, H0)
                if stochastic:
                    random_score = torch.rand_like(ei_score)
                    decisions = random_score < ei_score
                else:
                    decisions = ei_score > 0.5
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
                        g.add_edge(vi, g.vcount()-1)
                self._update_v(G, idx)

        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory
        return G

    def loss(self, mu, logvar, G_true, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._update_v(G, 0, H0)
        res = 0  # log likelihood
        for v_true in range(1, self.max_n):
            # calculate the likelihood of adding true types of nodes
            # use start type to denote padding vertices since start type only appears for vertex 0 
            # and will never be a true type for later vertices, thus it's free to use
            true_types = [g_true.vs[v_true]['type'] if v_true < g_true.vcount() 
                          else self.START_TYPE for g_true in G_true]
            Hg = self._get_graph_state(G, decode=True)
            type_scores = self.add_vertex(Hg)
            # vertex log likelihood
            vll = self.logsoftmax1(type_scores)[np.arange(len(G)), true_types].sum()  
            res = res + vll
            for i, g in enumerate(G):
                if true_types[i] != self.START_TYPE:
                    g.add_vertex(type=true_types[i])
            self._update_v(G, v_true)

            # calculate the likelihood of adding true edges
            true_edges = []
            for i, g_true in enumerate(G_true):
                true_edges.append(g_true.get_adjlist(igraph.IN)[v_true] if v_true < g_true.vcount()
                                  else [])
            edge_scores = []
            for vi in range(v_true-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, v_true)
                ei_score = self._get_edge_score(Hvi, H, H0)
                edge_scores.append(ei_score)
                for i, g in enumerate(G):
                    if vi in true_edges[i]:
                        g.add_edge(vi, v_true)
                self._update_v(G, v_true)
            edge_scores = torch.cat(edge_scores[::-1], 1)

            ground_truth = torch.zeros_like(edge_scores)
            idx1 = [i for i, x in enumerate(true_edges) for _ in range(len(x))]
            idx2 = [xx for x in true_edges for xx in x]
            ground_truth[idx1, idx2] = 1.0

            # edges log-likelihood
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum') 
            res = res + ell

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
        return loss
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G
    
class DVAE_DeepGMG(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, Td=3, Te=3, fast=False):
        super(DVAE_DeepGMG, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs * 2  # size of graph state
        self.bidir = bidirectional  # whether to ignore edge directions
        self.Td = Td  # message passing rounds when decoding
        self.Te = Te  # message passing rounds when encoding
        self.fast = fast
        self.device = None

        self.vs = hs

        # 0. encoding-related
        self.grue = nn.ModuleList()  # encoder GRU
        for t in range(self.Te):
            self.grue.append(nn.GRUCell(hs * 2, hs))
        self.fe = nn.Linear(self.hs * 2 + 1, self.hs * 2)
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar

        # initializing hidden state
        self.finit = nn.Linear(nvt + self.gs, hs)

        # 1. decoding-related
        self.grud = nn.ModuleList()  # decoder GRU
        for t in range(self.Td):
            self.grud.append(nn.GRUCell(hs * 2, hs))
        self.fc3 = nn.Linear(nz, self.gs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(self.gs, nvt),
                )  # which type of new vertex to add
        self.add_edge = nn.Sequential(
                nn.Linear(self.gs + hs, 1), 
                )  # whether to add edge
        self.select_node = nn.Sequential(
                nn.Linear(hs * 2, 1), 
                )  # select which node to connect from

        # 2. gate-related
        self.gate = nn.Sequential(
                nn.Linear(self.vs, self.gs), 
                nn.Sigmoid()
                )
        self.mapper = nn.Sequential(
                nn.Linear(self.vs, self.gs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.gate_init = nn.Sequential(
                nn.Linear(self.vs, self.gs), 
                nn.Sigmoid()
                )
        self.mapper_init = nn.Sequential(
                nn.Linear(self.vs, self.gs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor([[i] for i in idx])
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _initialize_v(self, G, v, H=None):
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if H is None:
            Hg = self._get_graph_state(G, 0, 1, init=True)  # exclude v itself
        else:  
            Hg = H
        Hv = self.finit(torch.cat([X, Hg], -1))
        for i, g in enumerate(G):
            g.vs[v]['H_forward'] = Hv[i:i+1]
        return

    def _propagate_to(self, G, v, propagator, t=0):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        Hv = self._get_vertex_state(G, v)
        if self.bidir:  # ignore edge directions, accept all neighbors' messages
            #H_nei = [[g.vs[x]['H_forward'] for x in g.neighbors(v)] for g in G]
            H_nei = [[g.vs[x]['H_forward'] for x in g.predecessors(v)] + [g.vs[x]['H_forward'] for x in g.successors(v)] for g in G]
            E_nei = [[1 for x in g.predecessors(v)] + [0 for x in g.successors(v)] for g in G]  # edge direction features
        else:  # only accept messages from predecessors (generalizing to directed cases)
            H_nei = [[g.vs[x]['H_forward'] for x in g.predecessors(v)] for g in G]
            E_nei = [[1 for x in g.predecessors(v)] for g in G]
            
        max_n_nei = max([len(x) for x in H_nei])  # maximum number of neighbors
        if max_n_nei == 0:
            for i, g in enumerate(G):
                g.vs[v]['H_tmp'] = Hv[i:i+1]
            return Hv
        mask = torch.ones(len(G), max_n_nei).to(self.get_device())
        for i, h_nei in enumerate(H_nei):
            if len(h_nei) < max_n_nei:
                mask[i][len(h_nei):] = 0
        H_nei = [torch.cat(h_nei + [self._get_zeros(max_n_nei - len(h_nei), self.hs)], 0).unsqueeze(0) 
                 for h_nei in H_nei]  # pad all to same length
        H_nei = torch.cat(H_nei, 0)  # batch * max_n_nei * hs
        E_nei = [torch.FloatTensor(e_nei + [0] * (max_n_nei - len(e_nei))).unsqueeze(0).unsqueeze(2).to(self.get_device())
                 for e_nei in E_nei]  # 
        E_nei = torch.cat(E_nei, 0)  # batch * max_n_nei * 1
        Hv_expand = Hv.unsqueeze(1).expand(-1, max_n_nei, -1)
        Av = self.fe(torch.cat([H_nei, E_nei, Hv_expand], -1))  # batch * max_n_nei * 2hs+1
        Av = Av * mask.unsqueeze(2).expand_as(Av)
        Av = Av.sum(1)  # batch * 2hs
        Hv = propagator[t](Av, Hv)  # batch * hs
        for i, g in enumerate(G):
            g.vs[v]['H_tmp'] = Hv[i:i+1]
        return Hv

    def _propagate(self, G, propagator, encoding=False):
        # do message passing for all nodes in all graphs in G
        if type(G) != list:
            G = [G]
        #prop_order = range(self.max_n)
        prop_order = range(max([g.vcount() for g in G]))
        T = self.Te if encoding else self.Td
        for t in range(T):
            for v_ in prop_order:
                self._propagate_to(G, v_, propagator, t)
            for g in G:
                #for v_ in range(g.vcount()):
                #    g.vs[v_]['H_forward'] = g.vs[v_]['H_tmp']
                g.vs['H_forward'] = g.vs['H_tmp']
        for g in G:
            del g.vs['H_tmp']  # delete tmp hidden states to save GPU memory
        return

    def encode(self, G):
        # encode graphs G into latent vectors
        # GCN propagation is now implemented in a non-parallel way for consistency, but
        # can definitely be parallel to speed it up. However, the major computation cost
        # comes from the generation, which is not parallellizable.
        if type(G) != list:
            G = [G]
        for v in range(self.max_n):
            self._initialize_v(G, v, torch.zeros(len(G), self.gs).to(self.get_device()))
        self._propagate(G, self.grue, True)
        Hg = self._get_graph_state(G) 
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, start=0, end_offset=0, init=False):
        # get the graph states, the R function
        Hg = []
        max_n_nodes = max(g.vcount() for g in G)
        for g in G:
            hg = [g.vs[i]['H_forward'] for i in range(start, g.vcount() - end_offset)]
            hg = torch.cat(hg, 0)
            hg = hg.unsqueeze(0)  # 1 * n * hs
            if g.vcount() < max_n_nodes:
                hg = torch.cat([hg, 
                    torch.zeros(1, max_n_nodes - g.vcount(), hg.shape[2]).to(self.get_device())],
                    1)  # 1 * max_n * hs
            Hg.append(hg)
        # gated sum node states as the graph state
        Hg = torch.cat(Hg, 0)  # batch * max_n * hs
        if not init:
            Hg = self._gated(Hg, self.gate, self.mapper).sum(1)  # batch * gs
        else:
            Hg = self._gated(Hg, self.gate_init, self.mapper_init).sum(1)  # batch * gs

        return Hg  # batch * gs

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, stochastic=True):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._initialize_v(G, 0, H0)
        finished = [False] * len(G)
        for idx in range(1, self.max_n):
            # decide the type of the next added vertex
            if idx == self.max_n - 1:  # force the last node to be end_type
                new_types = [self.END_TYPE] * len(G)
            else:
                self._propagate(G, self.grud)
                Hg = self._get_graph_state(G)
                type_scores = self.add_vertex(Hg)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) 
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            for i, g in enumerate(G):
                if not finished[i]:
                    g.add_vertex(type=new_types[i])
            self._initialize_v(G, idx)

            # decide connections
            edge_scores = []
            G_tmp = [g for i, g in enumerate(G) if not finished[i]]  # the remaining graphs to add new edges
            count = 0
            while G_tmp:
                count += 1
                if not self.fast:
                    self._propagate(G, self.grud)
                Hg = self._get_graph_state(G_tmp)
                H = self._get_vertex_state(G_tmp, idx)
                add_score = self.sigmoid(self.add_edge(torch.cat([Hg, H], 1)))  # |G_tmp| * 1
                if stochastic:
                    random_score = torch.rand_like(add_score)
                    decisions = random_score < add_score
                else:
                    decisions = add_score > 0.5
                G_tmp = [g for i, g in enumerate(G_tmp) if decisions[i, 0]]  # the remaining graphs to add new edges
                if not G_tmp or count > idx:
                    break

                Hv = torch.cat([self._get_vertex_state(G_tmp, v).unsqueeze(1) for v in range(idx)], 1)  # |G_tmp| * idx * hs
                H = self._get_vertex_state(G_tmp, idx).unsqueeze(1).expand_as(Hv)
                edge_score = self.select_node(torch.cat([Hv, H], -1)).squeeze(-1)  # |G_tmp| * idx
                if stochastic:
                    edge_prob = F.softmax(edge_score, 1).cpu().detach().numpy()
                    new_edge = [np.random.choice(range(idx), p=edge_prob[i]) 
                                 for i in range(len(G_tmp))]
                else:
                    new_edge = torch.argmax(edge_score, 1)
                    new_edge = new_edge.flatten().tolist()
                for i, g in enumerate(G_tmp):
                    g.add_edge(new_edge[i], g.vcount()-1)

            for i, g in enumerate(G):
                if not finished[i]:
                    if new_types[i] == self.END_TYPE: 
                        # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                            if v.index != g.vcount()-1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount()-1)
                        finished[i] = True
        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory
        return G

    def loss(self, mu, logvar, G_true, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._initialize_v(G, 0, H0)
        res = 0  # log likelihood
        for v_true in range(1, self.max_n):
            # calculate the likelihood of adding true types of nodes
            # use start type to denote padding vertices since start type only appears for vertex 0 
            # and will never be a true type for later vertices, thus it's free to use
            true_types = [g_true.vs[v_true]['type'] if v_true < g_true.vcount() 
                          else self.START_TYPE for g_true in G_true]
            self._propagate(G, self.grud)
            Hg = self._get_graph_state(G)
            type_scores = self.add_vertex(Hg)
            # vertex log likelihood
            vll = self.logsoftmax1(type_scores)[np.arange(len(G)), true_types].sum()  
            res = res + vll
            for i, g in enumerate(G):
                if true_types[i] != self.START_TYPE:
                    g.add_vertex(type=true_types[i])
            self._initialize_v(G, v_true)

            # calculate the likelihood of adding true edges
            true_edges = []
            for i, g_true in enumerate(G_true):
                true_edges.append(g_true.get_adjlist(igraph.IN)[v_true] if v_true < g_true.vcount()
                                  else [])
            graph_idx = range(len(G))
            while graph_idx:
                if not self.fast:
                    self._propagate(G, self.grud)
                Hg = self._get_graph_state(G)
                H = self._get_vertex_state(G, v_true)
                add_score = self.sigmoid(self.add_edge(torch.cat([Hg, H], 1)))[graph_idx]  # |graph_idx| * 1
                add_truth = [1.0 if true_edges[i] else 0.0 for i in graph_idx]
                add_truth = torch.tensor(add_truth).to(self.get_device())
                addll = -F.binary_cross_entropy(add_score.squeeze(1), add_truth, reduction='sum')

                Hv = torch.cat([self._get_vertex_state(G, v).unsqueeze(1) for v in range(v_true)], 1)  # batch * v_true * hs
                H = self._get_vertex_state(G, v_true).unsqueeze(1).expand_as(Hv)
                edge_score = self.select_node(torch.cat([Hv, H], -1)).squeeze(-1)  # batch * v_true
                edge_truth = []
                graph_idx = []  # graphs still having edges to add
                for i, g in enumerate(G):
                    if true_edges[i]:
                        u = true_edges[i].pop()
                        graph_idx.append(i)
                        edge_truth.append(u)
                        g.add_edge(u, v_true)
                ell = self.logsoftmax1(edge_score)[graph_idx, edge_truth].sum()  

                res = res + addll + ell

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
        return loss
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G
    
class DVAE_GCN(DVAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, levels=3):
        # bidirectional means passing messages ignoring edge directions
        super(DVAE_GCN, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.levels = levels
        self.gconv = nn.ModuleList()
        self.gconv.append(
                nn.Sequential(
                    nn.Linear(nvt, hs), 
                    nn.ReLU(), 
                    )
                )
        for lv in range(1, levels):
            self.gconv.append(
                    nn.Sequential(
                        nn.Linear(hs, hs), 
                        nn.ReLU(), 
                        )
                    )

    def _get_feature(self, g, v, lv=0):
        # get the node feature vector of v
        if lv == 0:  # initial level uses type features
            v_type = g.vs[v]['type']
            x = self._one_hot(v_type, self.nvt)
        else:
            x = g.vs[v]['H_forward']
        return x

    def _get_zero_x(self, n=1):
        # get zero predecessor states X, used for padding
        return torch.zeros(n, self.nvt).to(self.get_device())

    def _get_graph_state(self, G, decode=False, start=0, end_offset=0):
        # get the graph states
        # sum all node states between start and n-end_offset as the graph state
        Hg = []
        max_n_nodes = max(g.vcount() for g in G)
        for g in G:
            hg = torch.cat([g.vs[i]['H_forward'] for i in range(start, g.vcount() - end_offset)],
                           0).unsqueeze(0)  # 1 * n * hs
            if g.vcount() < max_n_nodes:
                hg = torch.cat([hg, 
                    torch.zeros(1, max_n_nodes - g.vcount(), hg.shape[2]).to(self.get_device())],
                    1)  # 1 * max_n * hs
            Hg.append(hg)
        # sum node states as the graph state
        Hg = torch.cat(Hg, 0).sum(1)  # batch * hs
        return Hg  # batch * hs

    def _GCN_propagate_to(self, G, v, lv=0):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return

        if self.bidir:  # ignore edge directions, accept all neighbors' messages
            H_nei = [[self._get_feature(g, v, lv)/(g.degree(v)+1)] + 
                     [self._get_feature(g, x, lv)/math.sqrt((g.degree(x)+1)*(g.degree(v)+1)) 
                     for x in g.neighbors(v)] for g in G]
        else:  # only accept messages from predecessors (generalizing GCN to directed cases)
            H_nei = [[self._get_feature(g, v, lv)/(g.indegree(v)+1)] + 
                     [self._get_feature(g, x, lv)/math.sqrt((g.outdegree(x)+1)*(g.indegree(v)+1)) 
                     for x in g.predecessors(v)] for g in G]
            
        max_n_nei = max([len(x) for x in H_nei])  # maximum number of neighbors
        H_nei = [torch.cat(h_nei + [self._get_zeros(max_n_nei - len(h_nei), h_nei[0].shape[1])], 0).unsqueeze(0) 
                 for h_nei in H_nei]  # pad all to same length
        H_nei = torch.cat(H_nei, 0)  # batch * max_n_nei * nvt
        Hv = self.gconv[lv](H_nei.sum(1))  # batch * hs
        for i, g in enumerate(G):
            g.vs[v]['H_forward'] = Hv[i:i+1]
        return Hv

    def encode(self, G):
        # encode graphs G into latent vectors
        # GCN propagation is now implemented in a non-parallel way for consistency, but
        # can definitely be parallel to speed it up. However, the major computation cost
        # comes from the generation, which is not parallellizable.
        if type(G) != list:
            G = [G]
        prop_order = range(self.max_n)
        for lv in range(self.levels):
            for v_ in prop_order:
                self._GCN_propagate_to(G, v_, lv)
        Hg = self._get_graph_state(G, start=1, end_offset=1)  # does not use the dummy input 
                                                              # and output nodes
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar
    
class SVAE(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False):
        super(SVAE, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.xs = (nvt + max_n-1)  # size of input x for GRU,
                                   # [one_hot(vertex_type), bit(connections)]
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # graph state size
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.device = None

        # 0. encoding-related
        self.grue = nn.GRU(self.xs, hs, batch_first=True, bidirectional=self.bidir)  # encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.GRU(hs, hs, batch_first=True)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, self.nvt), 
                )
        self.add_edges = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, self.max_n - 1), 
                )

        # 2. bidir-related, to unify sizes
        if self.bidir:
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.hs * 2, self.hs), 
                    )
            self.hv_unify = nn.Sequential(
                    nn.Linear(self.hs * 2, self.hs), 
                    )

        # 3. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _collate_fn(self, G):
        # create mini_batch of tensors from list G by padding
        # each graph g is a 1 * (n_vertex - 1) * (n_types + n_vertex-1) tensor
        # pad all to 1 * (max_n - 1) * (n_types + max_n-1) tensors
        if type(G) != list:
            G = [G]
        G_new = []
        for g in G:
            if g.shape[1] < self.max_n - 1:
                padding = torch.zeros(1, self.max_n-1-g.shape[1], g.shape[2]).to(self.get_device())
                padding[0, :, self.START_TYPE] = 1  # use start type's bit to indicate padding 
                                                    # nodes (since start types are never predicted)
                g = torch.cat([g, padding], 1)
            if g.shape[2] < self.xs:
                padding = torch.zeros(1, g.shape[1], self.xs-g.shape[2]).to(self.get_device())
                g = torch.cat([g, padding], 2)  # pad zeros to indicate no connections to padding 
                                                # nodes
            G_new.append(g)
        return torch.cat(G_new, 0)

    def encode(self, G):
        # G: [batch_size * max_n-1 * xs]
        _, Hn = self.grue(G)
        Hg = Hn.view(Hn.shape[1], -1)   # Hn's second dimension is "batch"
        if self.bidir:
            Hg = self.hg_unify(Hg)
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        #return mu
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _decode(self, z):
        H0 = self.relu(self.fc3(z))
        H_in = H0.unsqueeze(1).expand(-1, self.max_n - 1, -1)
        H_out, _ = self.grud(H_in)
        type_scores = self.add_vertex(H_out)  # batch * max_n-1 * nvt
        edge_scores = self.sigmoid(self.add_edges(H_out))  # batch * max_n-1 * max_n-1
        return type_scores, edge_scores

    def decode(self, z):
        type_scores, edge_scores = self._decode(z)
        return self.construct_igraph(type_scores, edge_scores)

    def loss(self, mu, logvar, G_true, beta=0.005):
        # G_true: [batch_size * max_n-1 * xs]
        z = self.reparameterize(mu, logvar)
        type_scores, edge_scores = self._decode(z)
        res = 0
        _, true_types = torch.max(G_true[:, :, :self.nvt], 2)
        res += F.cross_entropy(type_scores.transpose(1, 2), true_types, reduction='sum')
        true_edges = G_true[:, :, self.nvt:]
        res += F.binary_cross_entropy(edge_scores, true_edges, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def construct_igraph(self, type_scores, edge_scores, stochastic=True):
        # construct igraphs from node type and edge scores
        # note that when stochastic=True, type_scores should be raw scores before softmax, 
        # and edge_scores should probabilities between [0, 1] (after sigmoid)
        assert(type_scores.shape[:2] == edge_scores.shape[:2])
        if stochastic:
            type_probs = F.softmax(type_scores, 2).cpu().detach().numpy()
        G = []
        for gi in range(len(type_scores)):
            g = igraph.Graph(directed=True)
            g.add_vertex(type=self.START_TYPE)
            for vj in range(1, self.max_n):
                if vj == self.max_n - 1:
                    new_type = self.END_TYPE
                else:
                    if stochastic:
                        new_type = np.random.choice(range(self.nvt), p=type_probs[gi][vj-1])
                    else:
                        new_type = torch.argmax(type_scores[gi][vj-1], 0).item()
                g.add_vertex(type=new_type)
                if new_type == self.END_TYPE:  
                    end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                        if v.index != g.vcount()-1])
                    for v in end_vertices:
                        g.add_edge(v, vj)
                    break
                else:
                    for ek in range(vj):
                        ek_score = edge_scores[gi][vj-1][ek].item()
                        if stochastic:
                            if np.random.random_sample() < ek_score:
                                g.add_edge(ek, vj)
                        else:
                            if ek_score > 0.5:
                                g.add_edge(ek, vj)
            G.append(g)
        return G

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G