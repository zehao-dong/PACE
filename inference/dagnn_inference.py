import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import torch.nn.functional as F
from parameterLoader import argLoader
from models import PairWiseLearning, GraphEncoder
from utils.utils import floyed
from scipy import sparse
import igraph
from dagnn import DAGNN
import pickle
from tqdm import tqdm
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import shuffle
import scipy.io
import math
import pdb
import argparse
import networkx as nx
from torch_geometric.data import Data
from utils_dag import add_order_info
import time as TI

def generate_edges_from_adj(spr, start_idx=0):
    indic = spr.indices
    indpt = spr.indptr
    source = []
    tgt = []
    edge_count = 0
    for i in range(indpt.size-1):
        for j in range(indpt[i+1] - indpt[i]):
            source.append(i + start_idx)
            tgt.append(indic[edge_count] + start_idx)
            edge_count += 1
    return torch.LongTensor([source,tgt])


def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x

def decode_nas_to_pygraph(nodes, edges, adj_in, n_type=5, max_node=11):
    #n = len(nodes)  # add start_type 0, end_type  n_type + 1
    n = adj_in.shape[0]
    adj = np.zeros((max_node, max_node))
    adj_sub = np.zeros((n+2, n+2))
    
    x = []
    x += [one_hot(0,  n_type + 2)]
    # ignore start vertex
    node_types2 = []
    for i in range(n):
        x += [one_hot(nodes[i]+1, n_type + 2)]
        node_types2 += [nodes[i]+1]
    for i in range(max_node - n -1):
        x += [one_hot(n_type + 1, n_type + 2)]
        
    for i in range(edges.size(1)):
        adj[int(edges[0,i])+1, int(edges[1,i])+1] = 1
        adj_sub[int(edges[0,i])+1, int(edges[1,i])+1] = 1
    
    nx_graph_pre = nx.DiGraph(adj_sub)
    in_nodes = []
    out_nodes = []
    for i in range(n):
        if nx_graph_pre.in_degree(i+1) == 0:
            in_nodes.append(i+1)
        if nx_graph_pre.out_degree(i+1) == 0:
            out_nodes.append(i+1)
    del nx_graph_pre
    del adj_sub
    
    for i in in_nodes:
        adj[0,i] = 1
    for j in out_nodes:
        adj[j,n+1] = 1        
    # output node
    for k in range(n+1,max_node-1):
        adj[k, k + 1] = 1
        
    nx_graph = nx.DiGraph(adj)
    x = torch.cat(x, dim=0).float()
    edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous()
    # need "index" in name to be recognized during batch collation
    graph = Data(x=x, edge_index=edge_index)  # , edge_type=edge_type, maxy=m, y_index_0=torch.LongTensor(yi0), yi1=torch.LongTensor(yi1), y=torch.LongTensor(y))
    add_order_info(graph)
    # o = list(range(8))   #list(nx.topological_sort(nx_graph))
    # layers = torch.LongTensor([list(range(8)), o])
    # o.reverse()
    # layers2 = torch.LongTensor([list(range(8)), o])
    # to be able to use igraph methods in DVAE models
    graph.vs = [{'type': 0}] + [{'type': t} for t in node_types2] + [{'type': n_type+1}] * (max_node-n-1)
    return graph, n_type + 2

def generate_pyg_list(Xs, Rs, Acc, n_type, max_node):
    g_list = []
    pbar = tqdm(zip(Xs, Rs, Acc))
    for idx, (x, r, acc) in enumerate(pbar):
        adj = np.array(r)
        spr = sparse.csr_matrix(adj)
        edges = generate_edges_from_adj(spr)
        g, _ = decode_nas_to_pygraph(x,edges,adj, n_type,max_node)
        g_list.append((g,acc))
    return g_list

def prepare_graph(graph, config):
    if config.dataset.split('_')[0] == 'nasbench101':
        Xs, Rs, valid_accs, test_accs, times = zip(*graph)
        n_type = 5
        max_node = 11
    elif config.dataset.split('_')[0] == 'nasbench301':
        Xs, Rs, genos, predicted_accs, predicted_runtimes = zip(*graph)
        n_type = 11 # not add 2
        max_node = 17 # 15 + 2
    elif config.dataset.split('_')[0] == 'darts':
        Xs, Rs, genos = zip(*graph)
    elif config.dataset.split('_')[0] == 'oo':
        Xs, Rs, valid_accs, test_accs, times = zip(*graph)
    else:
        raise NotImplementedError()

    if config.dataset.split('_')[0] == 'nasbench101':
        return generate_pyg_list(Xs, Rs, test_accs, n_type, max_node), valid_accs, test_accs, times
    elif config.dataset.split('_')[0] == 'nasbench301':
        return generate_pyg_list(Xs, Rs, predicted_accs, n_type, max_node),  genos, predicted_accs, predicted_runtimes
    else:
        raise NotImplementedError()
    
        
def load_module_state(model, state_name):
    pretrained_dict = torch.load(state_name,map_location='cuda:2')
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return

config = argLoader()
args = argLoader()
torch.cuda.set_device(config.device)
INFER = True
RE = False
# Model
file_dir = os.path.dirname(os.path.realpath('__file__'))
res_dir = os.path.join(file_dir, 'results/{}{}'.format(config.dataset.split('_')[0], 'dagnn'))
if not os.path.exists(res_dir):
    os.makedirs(res_dir) 

if config.dataset.split('_')[0] == 'nasbench101':
    max_n = 11 # (9 + 2)
    num_vertex_type = 7 # (5 + 2)
    START_TYPE = 0
    END_TYPE = 6
elif config.dataset.split('_')[0] == 'nasbench301':
    max_n = 17 # 15 + 2
    num_vertex_type = 13 # 11 + 2
    START_TYPE = 0
    END_TYPE = 12


pkl_name = os.path.join(res_dir, config.dataset.split('_')[0] + 'glist' + '.pkl')        
model = DAGNN(
        num_vertex_type,
        301,
        301,
        max_n,
        num_vertex_type,
        START_TYPE,
        END_TYPE,
        num_nodes=max_n,
        hs=301, 
        nz=56, 
        bidirectional=False,
        agg='attn_h',
        num_layers=2,
        out_wx=False, 
        out_pool_all=False, 
        out_pool='max',
        dropout=0
        )

load_epoch = 30
load_module_state(model, os.path.join(res_dir, 
                                              'model_checkpoint{}.pth'.format(load_epoch)))

#net = PairWiseLearning(config)
if torch.cuda.is_available():
    model = model.cuda(config.device) 
model.eval()
    
data = []
trainSet = torch.load(config.train_data)
validSet = torch.load(config.valid_data)  
time_start=TI.time()
for dataset in [trainSet, validSet]:
    for i in range(len(dataset)):
        R = dataset[i]['adj']
        X = np.argmax(np.asarray(dataset[i]['ops']), axis=-1)
        if config.dataset.split('_')[0] == 'nasbench101':
            valid_acc = dataset[i]['validation_accuracy']
            test_acc = dataset[i]['test_accuracy']
            time = dataset[i]['training_time']
            data.append([X, R, valid_acc, test_acc, time])
        elif config.dataset.split('_')[0] == 'nasbench301':
            genotype = dataset[i]['genotype']
            predicted_acc = dataset[i]['predicted_acc']
            predicted_runtime = dataset[i]['predicted_runtime']
            data.append([X, R, genotype, predicted_acc, predicted_runtime])
        else:
            raise NotImplementedError()
if config.dataset.split('_')[0] == 'nasbench101':
    graph_list, valid_accs, test_accs, times = prepare_graph(data, config)
elif config.dataset.split('_')[0] == 'nasbench301':
    graph_list, genotypes, predicted_accs, predicted_runtimes = prepare_graph(data, config)
else:
    raise NotImplementedError()

embeddings = []
#time_start=TI.time()
with torch.no_grad():
    for i in range(0, len(data), 64):
        print('data {} / {}'.format(i, len(data)))
        bs = min(64, len(data) - i)
        g_batch = []
        for j in range(i,i+bs):
            g_batch.append(graph_list[j][0])  
        #g_batch = graph_list[i: i+bs]
        mu, logvar = model.encode(g_batch)
        embeddings.append(mu.cpu())
        g_batch = []
time_end=TI.time()
comp_time = time_end - time_start
embeddings = torch.cat(embeddings, dim=0)

if config.dataset.split('_')[0] == 'nasbench101':
    pretrained_embeddings = {'embeddings': embeddings, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
elif config.dataset.split('_')[0] == 'nasbench301':
    pretrained_embeddings = {'embeddings': embeddings, 'genotypes': genotypes, 'predicted_accs': predicted_accs, 'predicted_runtimes': predicted_runtimes}
else:
    raise NotImplementedError()

torch.save(pretrained_embeddings, 'dagnn_' + config.dataset + '.pt')
        