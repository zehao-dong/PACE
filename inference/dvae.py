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
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import shuffle
import scipy.io
import math
import pdb
import argparse
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
    pretrained_dict = torch.load(state_name)
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

def run_dvae(config):
    INFER = True
    RE = False
    CONT = True
    print(config.device)
    # Model
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    res_dir = os.path.join(file_dir, 'results/{}{}'.format(config.dataset.split('_')[0], 'dvae'))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir) 

    if config.dataset.split('_')[0] == 'nasbench101':
        max_n = 11
        num_vertex_type = 7
        START_TYPE = 0
        END_TYPE = 6
    elif config.dataset.split('_')[0] == 'nasbench301':
        max_n = 17 # 15 + 2
        num_vertex_type = 13 # 11 + 2
        START_TYPE = 0
        END_TYPE = 12

        
    pkl_name = os.path.join(res_dir, config.dataset.split('_')[0] + 'glist' + '.pkl')

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
            train_size_prop = int(len(trainSet) * 0.4)
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
    
    model = DVAE(
        max_n = max_n, 
        nvt = num_vertex_type, 
        START_TYPE = START_TYPE, 
        END_TYPE = END_TYPE, 
        hs=301, 
        nz=56, 
        bidirectional=False
        )

    #net = PairWiseLearning(config)
    if torch.cuda.is_available():
        model = model.cuda(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    
    if CONT:
        load_epoch = 40
        load_module_state(model, os.path.join(res_dir,'model_checkpoint{}.pth'.format(load_epoch)))
        load_module_state(optimizer, os.path.join(res_dir, 
                                                  'optimizer_checkpoint{}.pth'.format(load_epoch)))
        load_module_state(scheduler, os.path.join(res_dir, 
                                                  'scheduler_checkpoint{}.pth'.format(load_epoch)))
        
    min_loss = math.inf  # >= python 3.5
    min_loss_epoch = None
    loss_name = os.path.join(res_dir, 'train_loss.txt')
    loss_plot_name = os.path.join(res_dir, 'train_loss_plot.pdf')
    test_results_name = os.path.join(res_dir, 'test_results.txt')
    keep_old = False
    if os.path.exists(loss_name) and not keep_old:
        os.remove(loss_name)

    start_epoch = 40
    end_epoch = 40
    #if CONT:
    #    start_epoch += load_epoch
    #    end_epoch += load_epoch
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
    import time as TI
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
        time_start=TI.time()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(data), 64):
                print('data {} / {}'.format(i, len(data)))
                bs = min(64, len(data) - i)
                g_batch = []
                for j in range(i,i+bs):
                    g_batch.append(graph_list[j][0])
                mu, logvar = model.encode(g_batch)
                mu = mu.detach().cpu()
                logvar = logvar.detach().cpu()
                embeddings.append(mu)

        embeddings = torch.cat(embeddings, dim=0)
        time_end=TI.time()
        comp_time = time_end - time_start
        print('infer_time:', comp_time)
        if config.dataset.split('_')[0] == 'nasbench101':
            pretrained_embeddings = {'embeddings': embeddings, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
        elif config.dataset.split('_')[0] == 'nasbench301':
            pretrained_embeddings = {'embeddings': embeddings, 'genotypes': genotypes, 'predicted_accs': predicted_accs, 'predicted_runtimes': predicted_runtimes}
        else:
            raise NotImplementedError()

        torch.save(pretrained_embeddings, 'dvae_' + config.dataset + '.pt')

if __name__ == '__main__':
    args = argLoader()
    torch.cuda.set_device(args.device)
    run_dvae(args)