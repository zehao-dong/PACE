import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import torch.nn.functional as F
from parameterLoader import argLoader
from models.graphEncoder_pace import PairWiseLearning, GraphEncoder
from utils.utils import floyed
from scipy import sparse
import igraph
from g_encoders import DVAE
import pickle
from tqdm import tqdm
import time
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import shuffle
import scipy.io
import math
import pdb
import argparse
import igraph as ig
from scipy import sparse
import time as TI

def generate_edges_list_from_adj(spr):
    indic = spr.indices
    indpt = spr.indptr
    edges = []
    edge_count = 0
    for i in range(indpt.size-1):
        for j in range(indpt[i+1] - indpt[i]):
            source = i 
            tgt = int(indic[edge_count])
            edges.append((source,tgt))
            edge_count += 1
    return edges

"""
def get_canonical_label(x,r):
    adj = np.array(r)
    spr = sparse.csr_matrix(adj)
    edges = generate_edges_list_from_adj(spr)
    g = ig.Graph(directed=True)
    #num_node = len(x)
    num_node = adj.shape[0]
    g.add_vertices(num_node)
    g.add_edges(edges)
    c_label = g.canonical_permutation(color=x)
    return c_label
"""
def get_canonical_label(x,r):
    adj = np.array(r)
    spr = sparse.csr_matrix(adj)
    edges = generate_edges_list_from_adj(spr)
    g = ig.Graph(directed=True)
    #num_node = len(x)
    num_node = adj.shape[0]
    g.add_vertices(num_node)
    g.add_edges(edges)
    c_label = g.canonical_permutation(color=x[:num_node])
    clabel = c_label + [len(x)-1] * (len(x)-num_node)
    return clabel

def prepare_graph(graph, config):
    if config.dataset.split('_')[0] == 'nasbench101':
        Xs, Rs, valid_accs, test_accs, times = zip(*graph)
    elif config.dataset.split('_')[0] == 'nasbench301':
        Xs, Rs, genos, predicted_accs, predicted_runtimes = zip(*graph)
    elif config.dataset.split('_')[0] == 'darts':
        Xs, Rs, genos = zip(*graph)
    elif config.dataset.split('_')[0] == 'oo':
        Xs, Rs, valid_accs, test_accs, times = zip(*graph)
    else:
        raise NotImplementedError()

    ls = [len(it) for it in Xs]
    maxL = max(ls)
    inputs = []
    masks = []
    adjs = []
    clabels = []
    #ls = []
    for x, r, l in zip(Xs, Rs, ls):
    #for x, r in zip(Xs, Rs):
        input_i = torch.LongTensor(x)
        mask_i = torch.from_numpy(floyed(r)).float()
        adj_i = np.diag(np.ones(maxL)) 
        clabel_i = get_canonical_label(x,r)
        clabel_i = torch.LongTensor(clabel_i)
        r_np = np.array(r)
        #l = r_np.shape[0] # added
        num_node =r_np.shape[0] 
        adj_i[:num_node,:num_node] = r_np 
        adj_i = torch.from_numpy(adj_i).float()
        padded_input_i = F.pad(input_i, (0, maxL - l), "constant", config.PAD)
        padded_clabel_i = F.pad(clabel_i, (0, maxL - l), "constant", maxL-1)
        #padded_input_i = F.pad(input_i[:l], (0, maxL - l), "constant", config.PAD)
        #padded_clabel_i = F.pad(clabel_i[:l], (0, maxL - l), "constant", maxL-1)
        padded_mask_i = F.pad(mask_i, (0, maxL - mask_i.shape[1], 0, maxL - mask_i.shape[1]), "constant", config.PAD)
        inputs.append(padded_input_i)
        masks.append(padded_mask_i)
        adjs.append(adj_i)
        clabels.append(padded_clabel_i)
        #ls.append(l) # added

    if config.dataset.split('_')[0] == 'nasbench101':
        return torch.stack(inputs), torch.stack(masks), torch.stack(adjs),torch.stack(clabels),torch.LongTensor(ls),valid_accs, test_accs, times
    elif config.dataset.split('_')[0] == 'nasbench301':
        return torch.stack(inputs), torch.stack(masks), torch.stack(adjs), torch.stack(clabels),torch.LongTensor(ls), genos, predicted_accs, predicted_runtimes
    elif config.dataset.split('_')[0] == 'darts':
        return torch.stack(inputs), torch.stack(masks), torch.stack(clabels),torch.LongTensor(ls), torch.stack(adjs), genos
    elif config.dataset.split('_')[0] == 'oo':
        return torch.stack(inputs), torch.stack(masks), torch.stack(adjs), torch.stack(clabels),torch.LongTensor(ls), valid_accs, test_accs, times
    else:
        raise NotImplementedError()

def inference(config):
    # Model
    net = PairWiseLearning(config)
    if torch.cuda.is_available():
        net = net.cuda(config.device)

    # Convert DataParallel when testing
    pretrained_dict = torch.load(config.pretrained_path, map_location="cuda:{}".format(config.device))['state_dict']
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    net.load_state_dict(pretrained_dict)

    # Inference Parameters
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
            elif config.dataset.split('_')[0] == 'darts':
                genotype = dataset[i]['genotypes']
                data.append([X, R, genotype])
            elif config.dataset.split('_')[0] == 'oo':
                valid_acc = dataset[i]['validation_acc']
                test_acc = dataset[i]['test_acc']
                time = dataset[i]['training_time']
                data.append([X, R, valid_acc, test_acc, time])
            else:
                raise NotImplementedError()

    if config.dataset.split('_')[0] == 'nasbench101':
        X, maskX, adjX, clabelX, seq_l, valid_accs, test_accs, times = prepare_graph(data, config)
    elif config.dataset.split('_')[0] == 'nasbench301':
        X, maskX, adjX, clabelX, seq_l, genotypes, predicted_accs, predicted_runtimes = prepare_graph(data, config)
    elif config.dataset.split('_')[0] == 'darts':
        X, maskX, adjX, clabelX, seq_l,genotypes = prepare_graph(data, config)
    elif config.dataset.split('_')[0] == 'oo':
        X, maskX, adjX, clabelX, seq_l,valid_accs, test_accs, times = prepare_graph(data, config)
    else:
        raise NotImplementedError()

    maskX_ = maskX.transpose(-2, -1)

    # Inference
    net.eval()
    dropout = torch.nn.Dropout(p=config.dropout)
    dropout.eval()
    embeddings = []
    #time_start=TI.time()
    with torch.no_grad():
        for i in range(0, len(data), config.batch_size):
            print('data {} / {}'.format(i, len(data)))
            bs = min(args.batch_size, len(data) - i)
            x = X[i: i+bs].cuda(config.device)
            m = maskX[i: i+bs].cuda(config.device)
            m_ = maskX_[i: i+bs].cuda(config.device)
            adjx = adjX[i: i+bs].cuda(config.device)
            clabelx =  clabelX[i: i+bs].cuda(config.device)
            seq_ls = seq_l[i: i+bs].cuda(config.device)
            emb_x = dropout(net.opEmb(x))
            clemb_x = dropout(net.posEmb(clabelx))
            clemb_x = net.gnn_layer(clemb_x,adjx)
            h_x =  net.graph_encoder(emb_x+clemb_x, m, m_)
            #h_x =  net.graph_encoder(emb_x+clemb_x, m, m_)
            embeddings.append(GraphEncoder.get_embeddings(h_x))
            #embeddings.append(GraphEncoder.get_embeddings(h_x, seq_ls))
    time_end=TI.time()
    epoch_time = time_end - time_start
    print('inference time:', epoch_time)
    embeddings = torch.cat(embeddings, dim=0)
    if config.dataset.split('_')[0] == 'nasbench101':
        pretrained_embeddings = {'embeddings': embeddings, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
    elif config.dataset.split('_')[0] == 'nasbench301':
        pretrained_embeddings = {'embeddings': embeddings, 'genotypes': genotypes, 'predicted_accs': predicted_accs, 'predicted_runtimes': predicted_runtimes}
    elif config.dataset.split('_')[0] == 'darts':
        pretrained_embeddings = {'embeddings': embeddings, 'genotypes': genotypes}
    elif config.dataset.split('_')[0] == 'oo':
        pretrained_embeddings = {'embeddings': embeddings, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
    else:
        raise NotImplementedError()

    torch.save(pretrained_embeddings, 'pace_' + config.dataset + '.pt')

if __name__ == '__main__':
    args = argLoader()
    torch.cuda.set_device(args.device)
    inference(args)
        
        