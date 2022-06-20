from dagnn import DAGNN
from dagnn_bn import DAGNN_BN
from constants import *
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
import time
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

def run_dagnn(config):
    INFER = True
    RE = True
    CONT = False
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

        
    pkl_name = os.path.join(res_dir, config.dataset.split('_')[0] + 'pyglist' + '.pkl')

    # check whether to load pre-stored pickle data
    if os.path.isfile(pkl_name) and not RE:
        with open(pkl_name, 'rb') as f:
            train_graph_list, test_graph_list = pickle.load(f)
    # otherwise process the raw data and save to .pkl
    else:
        # Inference Parameters
        train_data = []
        test_data = []
        trainSet = torch.load(config.train_data)
        validSet = torch.load(config.valid_data)
        if config.dataset.split('_')[0] == 'nasbench301':
            train_size_prop = int(len(trainSet)*0.4)
            for i in range(train_size_prop):
                R = trainSet[i]['adj']
                X = np.argmax(np.asarray(trainSet[i]['ops']), axis=-1)
                if config.dataset.split('_')[0] == 'nasbench101':
                    valid_acc = trainSet[i]['validation_accuracy']
                    test_acc = trainSet[i]['test_accuracy']
                    ti = trainSet[i]['training_time']
                    train_data.append([X, R, valid_acc, test_acc, ti])
                elif config.dataset.split('_')[0] == 'nasbench301':
                    genotype = trainSet[i]['genotype']
                    predicted_acc = trainSet[i]['predicted_acc']
                    predicted_runtime = trainSet[i]['predicted_runtime']
                    train_data.append([X, R, genotype, predicted_acc, predicted_runtime])
                else:
                    raise NotImplementedError()
        else:
            for i in range(len(trainSet)):
                R = trainSet[i]['adj']
                X = np.argmax(np.asarray(trainSet[i]['ops']), axis=-1)
                if config.dataset.split('_')[0] == 'nasbench101':
                    valid_acc = trainSet[i]['validation_accuracy']
                    test_acc = trainSet[i]['test_accuracy']
                    ti = trainSet[i]['training_time']
                    train_data.append([X, R, valid_acc, test_acc, ti])
                elif config.dataset.split('_')[0] == 'nasbench301':
                    genotype = trainSet[i]['genotype']
                    predicted_acc = trainSet[i]['predicted_acc']
                    predicted_runtime = trainSet[i]['predicted_runtime']
                    train_data.append([X, R, genotype, predicted_acc, predicted_runtime])
                else:
                    raise NotImplementedError()

        for i in range(len(validSet)):
            R = validSet[i]['adj']
            X = np.argmax(np.asarray(validSet[i]['ops']), axis=-1)
            if config.dataset.split('_')[0] == 'nasbench101':
                valid_acc = validSet[i]['validation_accuracy']
                test_acc = validSet[i]['test_accuracy']
                ti = validSet[i]['training_time']
                test_data.append([X, R, valid_acc, test_acc, ti])
            elif config.dataset.split('_')[0] == 'nasbench301':
                genotype = validSet[i]['genotype']
                predicted_acc = validSet[i]['predicted_acc']
                predicted_runtime = validSet[i]['predicted_runtime']
                test_data.append([X, R, genotype, predicted_acc, predicted_runtime])
            else:
                raise NotImplementedError()

        if config.dataset.split('_')[0] == 'nasbench101':
            train_graph_list, _, _, _ = prepare_graph(train_data, config)
            test_graph_list, _, _, _  = prepare_graph(test_data, config)
        elif config.dataset.split('_')[0] == 'nasbench301':
            train_graph_list, _, _, _  = prepare_graph(train_data, config)
            test_graph_list, _, _, _  = prepare_graph(test_data, config)
        else:
            raise NotImplementedError()
        with open(pkl_name, 'wb') as f:
            pickle.dump((train_graph_list, test_graph_list), f)
    
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

    #net = PairWiseLearning(config)
    #if torch.cuda.is_available():
    model = model.cuda(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    
    if CONT:
        load_epoch = 20
        load_module_state(model, os.path.join(res_dir,'model_checkpoint{}.pth'.format(load_epoch)))
        load_module_state(optimizer, os.path.join(args.res_dir, 
                                                  'optimizer_checkpoint{}.pth'.format(load_epoch)))
        load_module_state(scheduler, os.path.join(args.res_dir, 
                                                  'scheduler_checkpoint{}.pth'.format(load_epoch)))
    min_loss = math.inf  # >= python 3.5
    min_loss_epoch = None
    loss_name = os.path.join(res_dir, 'train_loss.txt')
    loss_plot_name = os.path.join(res_dir, 'train_loss_plot.pdf')
    test_results_name = os.path.join(res_dir, 'test_results.txt')
    keep_old = False
    if os.path.exists(loss_name) and not keep_old:
        os.remove(loss_name)

    start_epoch = 1
    end_epoch = 31
    if CONT:
        start_epoch += load_epoch
        end_epoch += load_epoch
    import time
    for epoch in range(start_epoch, end_epoch):
        ### training
        time_start=time.time()
        model.train()
        train_loss = 0
        recon_loss = 0
        kld_loss = 0
        pred_loss = 0
        shuffle(train_graph_list)
        pbar = tqdm(train_graph_list)
        g_batch = []
        y_batch = []
        #time_start = time.time()
        for i, (g, y) in enumerate(pbar):
            g_batch.append(g)
            y_batch.append(y)
            if len(g_batch) == 64 or i == len(train_graph_list) - 1:
                optimizer.zero_grad()
                g_batch = model._collate_fn(g_batch)

                mu, logvar = model.encode(g_batch)
                loss, recon, kld = model.loss(mu, logvar, g_batch)
                pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                 epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                 kld.item()/len(g_batch)))
                loss.backward()

                train_loss += float(loss)
                recon_loss += float(recon)
                kld_loss += float(kld)
                optimizer.step()
                g_batch = []
                y_batch = []
        #time_end=time.time()
        time_end=time.time()
        comp_time = time_end - time_start
        print('====> Epoch: {} Average loss: {:.4f}, compute cost: {:.4f}'.format(
              epoch, train_loss / len(train_graph_list), comp_time))
        pred_loss = 0.0
        with open(loss_name, 'a') as loss_file:
            loss_file.write("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                train_loss/len(train_graph_list), 
                recon_loss/len(train_graph_list), 
                kld_loss/len(train_graph_list), 
                pred_loss/len(train_graph_list), 
                comp_time
                ))
        scheduler.step(train_loss)
        if epoch % 10 == 0:
            print("save current model...")
            model_name = os.path.join(res_dir, 'model_checkpoint{}.pth'.format(epoch))
            optimizer_name = os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
            scheduler_name = os.path.join(res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_name)
            torch.save(optimizer.state_dict(), optimizer_name)
            torch.save(scheduler.state_dict(), scheduler_name)
            
    #Nll, comp_time = test()
    
    if INFER:
        model.eval()
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
        elif config.dataset.split('_')[0] == 'nasbench301':
            graph_list, genotypes, predicted_accs, predicted_runtimes = prepare_graph(data, config)
        else:
            raise NotImplementedError()
            
        time_start=time.time()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(data), 64):
                print('data {} / {}'.format(i, len(data)))
                bs = min(64, len(data) - i)
                g_batch = []
                for j in range(i,i+bs):
                    g_batch.append(graph_list[j][0])  
                #g_batch = graph_list[i: i+bs]
                mu, logvar = model.encode(g_batch)
                embeddings.append(mu)
                g_batch = []
        time_end=time.time()
        comp_time = time_end - time_start
        print('infer_time:', comp_time)
        embeddings = torch.cat(embeddings, dim=0)

        if config.dataset.split('_')[0] == 'nasbench101':
            pretrained_embeddings = {'embeddings': embeddings, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
        elif config.dataset.split('_')[0] == 'nasbench301':
            pretrained_embeddings = {'embeddings': embeddings, 'genotypes': genotypes, 'predicted_accs': predicted_accs, 'predicted_runtimes': predicted_runtimes}
        else:
            raise NotImplementedError()

        torch.save(pretrained_embeddings, 'dagnn_' + config.dataset + '.pt')

if __name__ == '__main__':
    args = argLoader()
    torch.cuda.set_device(args.device)
    run_dagnn(args)