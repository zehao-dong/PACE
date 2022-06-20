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
from g_encoders import DVAE
import pickle
from tqdm import tqdm
import time as TI
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import shuffle
import scipy.io
import math
import pdb
import argparse

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


def decode_nas_to_igraph(nodes, edges, n_type, adj):
    n = adj.shape[0]
    g = igraph.Graph(directed=True)
    g.add_vertices(n+2)
    for i in range(n+2):
        if i == 0:
            g.vs[i]['type'] = 0
        elif i == n+1:
            g.vs[i]['type'] = n_type + 1
        else:
            g.vs[i]['type'] = nodes[i-1]+1
    for i in range(edges.size(1)):
        g.add_edge(int(edges[0,i])+1, int(edges[1,i])+1)  # always connect from last node ###################### change?
    in_vertices = [v.index for v in g.vs.select(_indegree_eq=0) if v.index != 0]
    for i in in_vertices:
        g.add_edge(0, i)
    end_vertices = [v.index for v in g.vs.select(_outdegree_eq=0) if v.index != n+1]
    for j in end_vertices:  # connect all loose-end vertices to the output node
        g.add_edge(j, n+1)
    g.topological_sorting()
    return g
"""
def decode_nas_to_igraph(nodes, edges, n_type):
    n = len(nodes)
    g = igraph.Graph(directed=True)
    g.add_vertices(n)
    g.vs['type'] = nodes
    for i in range(edges.size(1)):
        g.add_edge(int(edges[0,i]), int(edges[1,i]))  # always connect from last node ###################### change?
    return g
"""

def generate_graph_list(Xs, Rs, Acc, n_type):
    g_list = []
    pbar = tqdm(zip(Xs, Rs, Acc))
    for idx, (x, r, acc) in enumerate(pbar):
        adj = np.array(r)
        spr = sparse.csr_matrix(adj)
        edges = generate_edges_from_adj(spr)
        g = decode_nas_to_igraph(x,edges,n_type,adj)
        g_list.append((g,acc))
    return g_list

def prepare_graph(graph, config):
    if config.dataset.split('_')[0] == 'nasbench101':
        Xs, Rs, valid_accs, test_accs, times = zip(*graph)
        n_type = 5
    elif config.dataset.split('_')[0] == 'nasbench301':
        Xs, Rs, genos, predicted_accs, predicted_runtimes = zip(*graph)
        n_type = 9
    elif config.dataset.split('_')[0] == 'darts':
        Xs, Rs, genos = zip(*graph)
    elif config.dataset.split('_')[0] == 'oo':
        Xs, Rs, valid_accs, test_accs, times = zip(*graph)
    else:
        raise NotImplementedError()

    if config.dataset.split('_')[0] == 'nasbench101':
        return generate_graph_list(Xs, Rs, test_accs, n_type), valid_accs, test_accs, times
    elif config.dataset.split('_')[0] == 'nasbench301':
        return generate_graph_list(Xs, Rs, predicted_accs, n_type),  genos, predicted_accs, predicted_runtimes
    else:
        raise NotImplementedError()
        
def load_module_state(model, state_name):
    #pretrained_dict = torch.load(state_name,map_location='cuda:1')
    pretrained_dict = torch.load(state_name,map_location='cpu')
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


config = argLoader()
args = argLoader()
torch.cuda.set_device(config.device)
print(config.device)
INFER = True
RE = False
# Model
file_dir = os.path.dirname(os.path.realpath('__file__'))
res_dir = os.path.join(file_dir, 'results/{}{}'.format(config.dataset.split('_')[0], 'dvae'))
print(res_dir)
if not os.path.exists(res_dir):
    os.makedirs(res_dir) 

if config.dataset.split('_')[0] == 'nasbench101':
    max_n = 11
    num_vertex_type = 7 # 5 + 2
    START_TYPE = 0
    END_TYPE = 6
elif config.dataset.split('_')[0] == 'nasbench301':
    max_n = 17 # 15 + 2
    num_vertex_type = 13 # 11 + 2
    START_TYPE = 0
    END_TYPE = 12


#pkl_name = os.path.join(res_dir, config.dataset.split('_')[0] + 'glist' + '.pkl')

model = DVAE(
        max_n = max_n, 
        nvt = num_vertex_type, 
        START_TYPE = START_TYPE, 
        END_TYPE = END_TYPE, 
        hs=301, 
        nz=56, 
        bidirectional=False
        )
load_epoch = 20
load_module_state(model, os.path.join(res_dir, 
                                              'model_checkpoint{}.pth'.format(load_epoch)))

#device = torch.device("cuda:"+str(args.device))
#if torch.cuda.is_available():
#    model = model.to(device) 
model.eval()
#print(model.get_device())


pkl_name = os.path.join(res_dir, config.dataset.split('_')[0] + 'allglist' + '.pkl')

# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not RE:
    if config.dataset.split('_')[0] == 'nasbench101':
        with open(pkl_name, 'rb') as f:
            graph_list, valid_accs, test_accs, times = pickle.load(f)
    elif config.dataset.split('_')[0] == 'nasbench301':
        with open(pkl_name, 'rb') as f:
            graph_list, genotypes, predicted_accs, predicted_runtimes = pickle.load(f)
    else:
        raise NotImplementedError()
else:
    data = []
    trainSet = torch.load(config.train_data)
    validSet = torch.load(config.valid_data)  
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
        with open(pkl_name, 'wb') as f:
            pickle.dump((graph_list, valid_accs, test_accs, times), f)
    elif config.dataset.split('_')[0] == 'nasbench301':
        graph_list, genotypes, predicted_accs, predicted_runtimes = prepare_graph(data, config)
        with open(pkl_name, 'wb') as f:
            pickle.dump((graph_list, genotypes, predicted_accs, predicted_runtimes), f)
    else:
        raise NotImplementedError()

embeddings = []
time_start=TI.time()
#start=0
#end=350000

#start=350000
#end=700000

#start=700000
#end=1000000
from layers.graphEncoder_pace2 import GraphEncoder

with torch.no_grad():
    #for i in range(start, end, 64):
    for i in range(0, len(graph_list), 64):
        print('data {} / {}'.format(i, len(graph_list)))
        #if i != 121760:
        #bs = min(64, end-start-i)
        bs = min(64, len(graph_list)-i)
        g_batch = []
        for j in range(i,i+bs):
            g_batch.append(graph_list[j][0])  
        #g_batch = graph_list[i: i+bs]
        mu, logvar = model.encode(g_batch)
        embeddings.append(mu)
        #embeddings.append(GraphEncoder.get_embeddings2(mu))
        #del mu #delete unnecessary variables 
        #del logvar
        #gc.collect()
embeddings = torch.cat(embeddings, dim=0)
time_end=TI.time()
comp_time = time_end - time_start
print('infer_time:', comp_time)
#for indx in range(10):
#    embeddings = []
#    start = indx * len(graph_list) / 10
#    end = (indx+1) * len(graph_list) / 10
#    with torch.no_grad():
#        for i in range(int(start), int(end), 64):
#            print('data {} / {}'.format(i, len(graph_list)))
            #if i != 121760:
#            bs = min(64, len(graph_list) - i)
            #g_batch = []
            #for j in range(i,i+bs):
                #g_batch.append(graph_list[j][0])  
            #g_batch = graph_list[i: i+bs]
            #mu, logvar = model.encode(g_batch)
            #embeddings.append(mu.detach().cpu())
            #del mu
            #del logvar
            #g_batch = []        
    #embeddings = torch.cat(embeddings, dim=0)
if config.dataset.split('_')[0] == 'nasbench101':
            pretrained_embeddings = {'embeddings': embeddings, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
elif config.dataset.split('_')[0] == 'nasbench301':
    pretrained_embeddings = {'embeddings': embeddings, 'genotypes': genotypes, 'predicted_accs': predicted_accs, 'predicted_runtimes': predicted_runtimes}
else:
    raise NotImplementedError()

torch.save(pretrained_embeddings, 'dvae_' + config.dataset + '.pt')









        
        
