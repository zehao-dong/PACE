from __future__ import print_function
import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
import shutil
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util_bn import *
from models_bn import *
from bayesian_optimization.evaluate_BN import Eval_BN
import time


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='ENAS',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='DTRANS_VAE', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")

parser.add_argument('--predictor', action='store_true', default=False,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
parser.add_argument('--ninp', type=int, default=256, metavar='N',
                    help='position embedding and embedding size')
parser.add_argument('--nhid', type=int, default=512, metavar='N',
                    help='dimension of hidden state of transformer')
parser.add_argument('--nhead', type=int, default=8, metavar='N',
                    help='number of heads in self attention')
parser.add_argument('--nlayers', type=int, default=6, metavar='N',
                    help='number of self attention layers')
parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                    help='dropout rate in transformer')
parser.add_argument('--fc_hidden', type=int, default=32, metavar='N',
                    help='dropout rate in transformer')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')

# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--cuda_number', type=int, default=0,
                    help=' CUDA training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.cuda_number))
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, 
                                                                 args.save_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data, graph_args = pickle.load(f)
# otherwise process the raw data and save to .pkl
else:
    # determine data formats according to models, DVAE: igraph, SVAE: string (as tensors)
    if args.model.startswith('DTRANS_VAE'):
        input_fmt = 'igraph'
    elif args.model.startswith('SVAE'):
        input_fmt = 'string'
    if args.data_type == 'ENAS':
        train_data, test_data, graph_args = load_ENAS_graphs(args.data_name, n_types=args.nvt,
                                                             fmt=input_fmt)
    elif args.data_type == 'BN':
        train_data, test_data, graph_args = load_BN_graphs(args.data_name, n_types=args.nvt,
                                                           fmt=input_fmt)
    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data, graph_args), f)

# delete old files in the result directory
remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and 
        not f.startswith('train_graph') and not f.startswith('test_graph') and
        not f.endswith('.pth')]
for f in remove_list:
    tmp = os.path.join(args.res_dir, f)
    if not os.path.isdir(tmp) and not args.keep_old:
        os.remove(tmp)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

# construct train data
if args.no_test:
    train_data = train_data + test_data

if args.small_train:
    train_data = train_data[:100]


'''Prepare the model'''
# model
model = PACE_VAE_nodagseq(
        max_n = graph_args.max_n, 
        nvt = graph_args.num_vertex_type, 
        START_TYPE = graph_args.START_TYPE, 
        END_TYPE = graph_args.END_TYPE, 
        START_SYMBOL = graph_args.START_SYMBOL,
        ninp = args.ninp,
        nhead = args.nhead,
        nhid = args.nhid,
        nlayers=args.nlayers, 
        dropout=args.dropout, 
        fc_hidden=args.fc_hidden,
        nz = args.nz
        )

if args.predictor:
    pred_size = (graph_args.max_n-1) * args.nhid
    predictor = nn.Sequential(
            nn.Linear(pred_size, args.nhid), 
            nn.Tanh(), 
            nn.Linear(args.nhid, 1)
            )
    model.predictor = predictor
    model.mseloss = nn.MSELoss(reduction='sum')
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)

if args.all_gpus:
    net = custom_DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'))
else:
    if args.continue_from is not None:
        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 
                                              'model_checkpoint{}.pth'.format(epoch)))
        load_module_state(optimizer, os.path.join(args.res_dir, 
                                                  'optimizer_checkpoint{}.pth'.format(epoch)))
        load_module_state(scheduler, os.path.join(args.res_dir, 
                                                  'scheduler_checkpoint{}.pth'.format(epoch)))

# plot sample train/test graphs
if not os.path.exists(os.path.join(args.res_dir, 'train_graph_id0.pdf')) or args.reprocess:
    if not args.keep_old:
        for data in ['train_data', 'test_data']:
            G = [g for g, y in eval(data)[:10]]
            if args.model.startswith('SVAE'):
                G = [g.to(device) for g in G]
                G = model._collate_fn(G)
                G = model.construct_igraph(G[:, :, :model.nvt], G[:, :, model.nvt:], False)
            for i, g in enumerate(G):
                name = '{}_graph_id{}'.format(data[:-5], i)
                plot_DAG(g, args.res_dir, name, data_type=args.data_type)


'''Define some train/test functions'''
def train(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pred_loss = 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    y_batch = []
    time_start=time.time()
    for i, (g, y) in enumerate(pbar):
        if args.model.startswith('SVAE'):  # for SVAE, g is tensor
            g = g.to(device)
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            if args.all_gpus:  # does not support predictor yet
                loss = net(g_batch).sum()
                pbar.set_description('Epoch: %d, loss: %0.4f' % (epoch, loss.item()/len(g_batch)))
                recon, kld = 0, 0
            else:
                mu, logvar = model.encode(g_batch)
                loss, recon, kld = model.loss(mu, logvar, g_batch)
                if args.predictor:
                    y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                    y_pred = model.predictor(mu)
                    pred = model.mseloss(y_pred, y_batch)
                    loss += pred
                    pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f, pred: %0.4f'\
                            % (epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                            kld.item()/len(g_batch), pred/len(g_batch)))
                else:
                    pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                     epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                     kld.item()/len(g_batch)))
            loss.backward()
            
            train_loss += float(loss)
            if args.predictor:
                pred_loss += float(pred_loss)
            optimizer.step()
            g_batch = []
            y_batch = []
    time_end=time.time()
    comp_time =time_end-time_start
    print('====> Epoch: {0} Average loss: {1:.4f}, Average compute time: {2:.4f}'.format(
          epoch, train_loss / len(train_data), comp_time/ len(train_data)))

    if args.predictor:
        return train_loss, recon_loss, kld_loss, pred_loss, comp_time
    return train_loss, recon_loss, kld_loss, comp_time


def visualize_recon(epoch):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    for i, (g, y) in enumerate(test_data[:10]+train_data[:10]):
        if args.model.startswith('SVAE'):
            g = g.to(device)
            g = model._collate_fn(g)
            g_recon = model.encode_decode(g)[0]
            g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)[0]
        elif args.model.startswith('DVAE'):
            g_recon = model.encode_decode(g)[0]
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)
        plot_DAG(g, args.res_dir, name0, data_type=args.data_type)
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        plot_DAG(g_recon, args.res_dir, name1, data_type=args.data_type)


def test():
    # test recon accuracy
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    n_perfect = 0
    print('Testing begins...')
    pbar = tqdm(test_data)
    g_batch = []
    y_batch = []
    time_start=time.time()
    for i, (g, y) in enumerate(pbar):
        if args.model.startswith('SVAE'):
            g = g.to(device)
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g = model._collate_fn(g_batch)
            mu, logvar = model.encode(g)
            _, nll, _ = model.loss(mu, logvar, g)
            pbar.set_description('nll: {:.4f}'.format(nll.item()/len(g_batch)))
            Nll += nll.item()
            if args.predictor:
                y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                y_pred = model.predictor(mu)
                pred = model.mseloss(y_pred, y_batch)
                pred_loss += pred.item()
            # construct igraph g from tensor g to check recon quality
            if args.model.startswith('SVAE'):  
                g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)
            for _ in range(encode_times):
                z = model.reparameterize(mu, logvar)
                for _ in range(decode_times):
                    g_recon = model.decode(z)
                    n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))
            g_batch = []
            y_batch = []
    time_end=time.time()
    comp_time = time_end - time_start
    Nll /= len(test_data)
    pred_loss /= len(test_data)
    pred_rmse = math.sqrt(pred_loss)
    acc = n_perfect / (len(test_data) * encode_times * decode_times)
    if args.predictor:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}, pred rmse: {2:.4f}, compute time cost: {3:.4f}'.format(
            Nll, acc, pred_rmse, comp_time/len(test_data)))
        return Nll, acc, pred_rmse, comp_time
    else:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}, compute time cost: {2: .4f}'.format(Nll, acc, comp_time/len(test_data)))
        return Nll, acc, comp_time


def prior_validity(scale_to_train_range=False):
    if scale_to_train_range:
        Z_train, Y_train = extract_latent(train_data)
        z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    n_latent_points = 1000
    decode_times = 10
    n_valid = 0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    G_train = [g for g, y in train_data]
    if args.model.startswith('SVAE'):
        G_train = [g.to(device) for g in G_train]
        G_train = model._collate_fn(G_train)
        G_train = model.construct_igraph(G_train[:, :, :model.nvt], G_train[:, :, model.nvt:], False)
    pbar = tqdm(range(n_latent_points))
    cnt = 0
    for i in pbar:
        cnt += 1
        if cnt == args.infer_batch_size or i == n_latent_points - 1:
            z = torch.randn(cnt, model.nz).to(model._get_device())
            if scale_to_train_range:
                z = z * z_std + z_mean  # move to train's latent range
            for j in range(decode_times):
                g_batch = model.decode(z)
                G.extend(g_batch)
                if args.data_type == 'ENAS':
                    for g in g_batch:
                        if is_valid_ENAS(g, graph_args.START_TYPE, graph_args.END_TYPE):
                            n_valid += 1
                            G_valid.append(g)
                elif args.data_type == 'BN':
                    for g in g_batch:
                        if is_valid_BN(g, graph_args.START_TYPE, graph_args.END_TYPE):
                            n_valid += 1
                            G_valid.append(g)
            cnt = 0
    r_valid = n_valid / (n_latent_points * decode_times)
    print('Ratio of valid decodings from the prior: {:.4f}'.format(r_valid))

    G_valid_str = [decode_igraph_to_ENAS(g) for g in G_valid]
    r_unique = len(set(G_valid_str)) / len(G_valid_str) if len(G_valid_str)!=0 else 0.0
    print('Ratio of unique decodings from the prior: {:.4f}'.format(r_unique))

    r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))
    return r_valid, r_unique, r_novel


def extract_latent(data):
    model.eval()
    Z = []
    Y = []
    g_batch = []
    for i, (g, y) in enumerate(tqdm(data)):
        if args.model.startswith('SVAE'):
            g_ = g.to(device)
        elif args.model.startswith('DTRANS_VAE'):
            # copy igraph
            # otherwise original igraphs will save the H states and consume more GPU memory
            g_ = g.copy()  
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
        Y.append(y)
    return np.concatenate(Z, 0), np.array(Y)


'''Extract latent representations Z'''
def save_latent_representations(epoch):
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test
                         }
                     )


def interpolation_exp(epoch, num=5):
    print('Interpolation experiments between two random testing graphs')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    if args.data_type == 'BN':
        eva = Eval_BN(interpolation_res_dir)
    interpolate_number = 10
    model.eval()
    cnt = 0
    for i in range(0, len(test_data), 2):
        cnt += 1
        (g0, _), (g1, _) = test_data[i], test_data[i+1]
        if args.model.startswith('SVAE'):
            g0 = g0.to(device)
            g1 = g1.to(device)
            g0 = model._collate_fn([g0])
            g1 = model._collate_fn([g1])
        z0, _ = model.encode(g0)
        z1, _ = model.encode(g1)
        print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
        print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
        Z = []  # to store all the interpolation points
        for j in range(0, interpolate_number + 1):
            zj = z0 + (z1 - z0) / interpolate_number * j
            Z.append(zj)
        Z = torch.cat(Z, 0)
        # decode many times and select the most common one
        G, G_str = decode_from_latent_space(Z, model, return_igraph=True,
                                            data_type=args.data_type) 
        names = []
        scores = []
        for j in range(0, interpolate_number + 1):
            namej = 'graph_interpolate_{}_{}_of_{}'.format(i, j, interpolate_number)
            namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                             data_type=args.data_type)
            names.append(namej)
            if args.data_type == 'BN':
                scorej = eva.eval(G_str[j])
                scores.append(scorej)
        fig = plt.figure(figsize=(120, 20))
        for j, namej in enumerate(names):
            imgj = mpimg.imread(namej)
            fig.add_subplot(1, interpolate_number + 1, j + 1)
            plt.imshow(imgj)
            if args.data_type == 'BN':
                plt.title('{}'.format(scores[j]), fontsize=40)
            plt.axis('off')
        plt.savefig(os.path.join(args.res_dir, 
                    args.data_name + '_{}_interpolate_exp_ensemble_epoch{}_{}.pdf'.format(
                    args.model, epoch, i)), bbox_inches='tight')
        '''
        # draw figures with the same height
        new_name = os.path.join(args.res_dir, args.data_name + 
                                '_{}_interpolate_exp_ensemble_{}.pdf'.format(args.model, i))
        combine_figs_horizontally(names, new_name)
        '''
        if cnt == num:
            break


def interpolation_exp2(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments between flat-net and dense-net')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation2')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 10
    model.eval()
    n = graph_args.max_n
    g0 = [[1]+[0]*(i-1) for i in range(1, n-1)]  # this is flat-net
    g1 = [[1]+[1]*(i-1) for i in range(1, n-1)]  # this is dense-net

    if args.model.startswith('SVAE'):
        g0, _ = decode_ENAS_to_tensor(str(g0), n_types=6)
        g1, _ = decode_ENAS_to_tensor(str(g1), n_types=6)
        g0 = g0.to(device)
        g1 = g1.to(device)
        g0 = model._collate_fn([g0])
        g1 = model._collate_fn([g1])
    elif args.model.startswith('DVAE'):
        g0, _ = decode_ENAS_to_igraph(str(g0))
        g1, _ = decode_ENAS_to_igraph(str(g1))
    z0, _ = model.encode(g0)
    z1, _ = model.encode(g1)
    print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        zj = z0 + (z1 - z0) / interpolate_number * j
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)  
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                         data_type=args.data_type)
        names.append(namej)
    fig = plt.figure(figsize=(120, 20))
    for j, namej in enumerate(names):
        imgj = mpimg.imread(namej)
        fig.add_subplot(1, interpolate_number + 1, j + 1)
        plt.imshow(imgj)
        plt.axis('off')
    plt.savefig(os.path.join(args.res_dir, 
                args.data_name + '_{}_interpolate_exp2_ensemble_epoch{}.pdf'.format(
                args.model, epoch)), bbox_inches='tight')


def interpolation_exp3(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments around a great circle')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation3')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 36
    model.eval()
    n = graph_args.max_n
    g0 = [[1]+[0]*(i-1) for i in range(1, n-1)]  # this is flat-net
    if args.model.startswith('SVAE'):
        g0, _ = decode_ENAS_to_tensor(str(g0), n_types=6)
        g0 = g0.to(device)
        g0 = model._collate_fn([g0])
    elif args.model.startswith('DVAE'):
        g0, _ = decode_ENAS_to_igraph(str(g0))
    z0, _ = model.encode(g0)
    norm0 = torch.norm(z0)
    z1 = torch.ones_like(z0)
    # there are infinite possible directions that are orthogonal to z0,
    # we just randomly pick one from a finite set
    dim_to_change = random.randint(0, z0.shape[1]-1)  # this to get different great circles
    print(dim_to_change)
    z1[0, dim_to_change] = -(z0[0, :].sum() - z0[0, dim_to_change]) / z0[0, dim_to_change]
    z1 = z1 / torch.norm(z1) * norm0
    print('z0: ', z0, 'z1: ', z1, 'dot product: ', (z0 * z1).sum().item())
    print('norm of z0: {}, norm of z1: {}'.format(norm0, torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    omega = torch.acos(torch.dot(z0.flatten(), z1.flatten()))
    print('angle between z0 and z1: {}'.format(omega))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        theta = 2*math.pi / interpolate_number * j
        zj = z0 * np.cos(theta) + z1 * np.sin(theta)
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type) 
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                         data_type=args.data_type)
        names.append(namej)
    # draw figures with the same height
    new_name = os.path.join(args.res_dir, args.data_name + 
                            '_{}_interpolate_exp3_ensemble_epoch{}.pdf'.format(args.model, epoch))
    combine_figs_horizontally(names, new_name)


def smoothness_exp(epoch, gap=0.05):
    print('Smoothness experiments around a latent vector')
    smoothness_res_dir = os.path.join(args.res_dir, 'smoothness')
    if not os.path.exists(smoothness_res_dir):
        os.makedirs(smoothness_res_dir) 
    
    #z0 = torch.zeros(1, model.nz).to(device)  # use all-zero vector as center
    
    if args.data_type == 'ENAS': 
        g_str = '4 4 0 3 0 0 5 0 0 1 2 0 0 0 0 5 0 0 0 1 0'  # a 6-layer network
        row = [int(x) for x in g_str.split()]
        row = flat_ENAS_to_nested(row, model.max_n-2)
        if args.model.startswith('SVAE'):
            g0, _ = decode_ENAS_to_tensor(row, n_types=model.max_n-2)
            g0 = g0.to(device)
            g0 = model._collate_fn([g0])
        elif args.model.startswith('DVAE'):
            g0, _ = decode_ENAS_to_igraph(row)
    elif args.data_type == 'BN':
        g0 = train_data[20][0]
        if args.model.startswith('SVAE'):
            g0 = g0.to(device)
            g0 = model._collate_fn([g0])
    z0, _ = model.encode(g0)

    # select two orthogonal directions in latent space
    tmp = np.random.randn(z0.shape[1], z0.shape[1])
    Q, R = qr(tmp)
    dir1 = torch.FloatTensor(tmp[0:1, :]).to(device)
    dir2 = torch.FloatTensor(tmp[1:2, :]).to(device)

    # generate architectures along two orthogonal directions
    grid_size = 13
    grid_size = 9
    mid = grid_size // 2
    Z = []
    pbar = tqdm(range(grid_size ** 2))
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        zij = z0 + dir1 * (i - mid) * gap + dir2 * (j - mid) * gap
        Z.append(zij)
    Z = torch.cat(Z, 0)
    if True:
        G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)
    else:  # decode by 3 batches in case of GPU out of memory 
        Z0, Z1, Z2 = Z[:len(Z)//3, :], Z[len(Z)//3:len(Z)//3*2, :], Z[len(Z)//3*2:, :]
        G = []
        G += decode_from_latent_space(Z0, model, return_igraph=True, data_type=args.data_type)[0]
        G += decode_from_latent_space(Z1, model, return_igraph=True, data_type=args.data_type)[0]
        G += decode_from_latent_space(Z2, model, return_igraph=True, data_type=args.data_type)[0]
    names = []
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        pbar.set_description('Drawing row {}/{}, col {}/{}...'.format(i+1, 
                             grid_size, j+1, grid_size))
        nameij = 'graph_smoothness{}_{}'.format(i, j)
        nameij = plot_DAG(G[idx], smoothness_res_dir, nameij, data_type=args.data_type)
        names.append(nameij)
    #fig = plt.figure(figsize=(200, 200))
    if args.data_type == 'ENAS':
        fig = plt.figure(figsize=(50, 50))
    elif args.data_type == 'BN':
        fig = plt.figure(figsize=(30, 30))
    
    nrow, ncol = grid_size, grid_size
    for ij, nameij in enumerate(names):
        imgij = mpimg.imread(nameij)
        fig.add_subplot(nrow, ncol, ij + 1)
        plt.imshow(imgij)
        plt.axis('off')
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.savefig(os.path.join(args.res_dir, 
                args.data_name + '_{}_smoothness_ensemble_epoch{}_gap={}_small.pdf'.format(
                args.model, epoch, gap)), bbox_inches='tight')


'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

if args.only_test:
    epoch = args.continue_from
    #sampled = model.generate_sample(args.sample_number)
    #save_latent_representations(epoch)
    visualize_recon(300)
    #interpolation_exp2(epoch)
    #interpolation_exp3(epoch)
    #prior_validity(True)
    #test()
    #smoothness_exp(epoch, 0.1)
    #smoothness_exp(epoch, 0.05)
    #interpolation_exp(epoch)
    pdb.set_trace()

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    if args.predictor:
        train_loss, recon_loss, kld_loss, pred_loss,comp_time = train(epoch)
    else:
        train_loss, recon_loss, kld_loss,comp_time = train(epoch)
        pred_loss = 0.0
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
            train_loss/len(train_data), 
            recon_loss/len(train_data), 
            kld_loss/len(train_data), 
            pred_loss/len(train_data), 
            comp_time/len(train_data)
            ))
    scheduler.step(train_loss)
    if epoch % args.save_interval == 0:
        print("save current model...")
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)
        #print("visualize reconstruction examples...")
        #visualize_recon(epoch)
        print("extract latent representations...")
        save_latent_representations(epoch)
        #print("sample from prior...")
        #sampled = model.generate_sample(args.sample_number)
        #for i, g in enumerate(sampled):
        #    namei = 'graph_{}_sample{}'.format(epoch, i)
        #    plot_DAG(g, args.res_dir, namei, data_type=args.data_type)
        print("plot train loss...")
        losses = np.loadtxt(loss_name)
        if losses.ndim == 1:
            continue
        fig = plt.figure()
        num_points = losses.shape[0]
        plt.plot(range(1, num_points+1), losses[:, 0], label='Total')
        plt.plot(range(1, num_points+1), losses[:, 1], label='Recon')
        plt.plot(range(1, num_points+1), losses[:, 2], label='KLD')
        plt.plot(range(1, num_points+1), losses[:, 3], label='Pred')
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.legend()
        plt.savefig(loss_plot_name)

'''Testing begins here'''
if args.predictor:
    Nll, acc, pred_rmse, comp_time = test()
else:
    Nll, acc, comp_time = test()
    pred_rmse = 0
r_valid, r_unique, r_novel = prior_validity(True)
with open(test_results_name, 'a') as result_file:
    result_file.write("Epoch {} Test recon loss: {} recon acc: {:.4f} r_valid: {:.4f}".format(
            epoch, Nll, acc, r_valid) + 
            " r_unique: {:.4f}  Compute_cost:P{:.4f} pred_rmse: {:.4f}\n".format(r_unique, comp_time,
            pred_rmse))
#interpolation_exp2(epoch)
class PACE_nodagseq(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, START_SYMBOL, ninp=256, nhead=8, nhid=512, nlayers=6, dropout=0.25, fc_hidden=256, nz = 64):
        super(PACE_nodagseq,self).__init__()
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
        
        hidden_size = self.nhid*self.max_n
        self.hidden_size = hidden_size
        #self.fc1 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        self.fc1 = nn.Linear(hidden_size,nz)
        self.fc2 = nn.Linear(hidden_size,nz)
        #nn.Linear(hidden_size,nz) 
        #self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz))  # latent logvar

        # 2. decoder-related
        decoder_layers = TransformerDecoder_layer(nhid, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers,nlayers)

        self.add_node = nn.Sequential(
            nn.Linear(nhid,fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, nvt)
            )
        self.add_edge = nn.Sequential(
                nn.Linear(nhid * 2, nhid), 
                nn.ReLU(), 
                nn.Linear(nhid, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew

        #self.fc3 = nn.Sequential(
        #    nn.Linear(nz,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,hidden_size))
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
            ptype = g.vs['position']
            num_node = len(ntype)
            if num_node < self.max_n:
                ntype += [self.END_TYPE] * (self.max_n - num_node)
                ptype += [max(ptype)+1] * (self.max_n - num_node)

            # node i feature
            ntype_one_hot = self._one_hot(ntype,self.nvt)
            node_feature[i,:,:] = ntype_one_hot # the 'extra' nodes are padded with the zero embeddings
            # position one-hot
            pos_one_hot[i,:,:] = self._one_hot(ptype,self.max_n)
            # node i adj
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
            head_count += self.nhead
        return node_feature, pos_one_hot, adj, src_mask.to(torch.bool), tgt_mask.to(torch.bool).transpose(1,2), mem_mask.to(torch.bool), graph_sizes, true_types

    def encode(self, glist):
        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        bsize = len(graph_sizes) 

        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        # here we set the source sequence and the tgt sequence for the teacher forcing
        src_inp = node_feat.transpose(0,1) # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)

        
        #memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp,mask=tgt_mask)
        memory = memory.transpose(0,1).reshape(-1,self.max_n*self.nhid) # shape ( bsize, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.

        return self.fc1(memory), self.fc2(memory)

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
        memory = self.fc3(z).reshape(-1,self.max_n,self.nhid).transpose(0,1)

        G = [igraph.Graph(directed=True) for _ in range(bsize)]
        for g in G:
            g.add_vertex(type=self.START_SYMBOL)
            g.add_vertex(type=self.START_TYPE)
            g.add_edge(0,1)
            g.vs[0]['position'] = 0
            g.vs[1]['position'] = 1

        #memory = self.encoder(src_inp,mask=src_mask)

        finished = [False] * bsize
        for idx in range(2, self.max_n): # the first two type of nodes are certain
            node_one_hot, pos_one_hot, adj, _, tgt_mask, mem_mask, _, _ = self._prepare_features(G,self.max_n-1)
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
        memory = self.fc3(z).reshape(-1,self.max_n,self.nhid).transpose(0,1)

        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
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

    
class PACE_mask(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, START_SYMBOL, ninp=256, nhead=8, nhid=512, nlayers=6, dropout=0.25, fc_hidden=256, nz = 64):
        super(PACE_mask,self).__init__()
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
        
        hidden_size = self.nhid*self.max_n
        self.hidden_size = hidden_size
        #self.fc1 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        self.fc1 = nn.Linear(hidden_size,nz)
        self.fc2 = nn.Linear(hidden_size,nz)
        #nn.Linear(hidden_size,nz) 
        #self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz))  # latent logvar

        # 2. decoder-related
        decoder_layers = TransformerDecoder_layer(nhid, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers,nlayers)

        self.add_node = nn.Sequential(
            nn.Linear(nhid,fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, nvt)
            )
        self.add_edge = nn.Sequential(
                nn.Linear(nhid * 2, nhid), 
                nn.ReLU(), 
                nn.Linear(nhid, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew

        #self.fc3 = nn.Sequential(
        #    nn.Linear(nz,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,hidden_size))
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
            ptype = g.vs['position']
            num_node = len(ntype)
            if num_node < self.max_n:
                ntype += [self.END_TYPE] * (self.max_n - num_node)
                ptype += [max(ptype)+1] * (self.max_n - num_node)

            # node i feature
            ntype_one_hot = self._one_hot(ntype,self.nvt)
            node_feature[i,:,:] = ntype_one_hot # the 'extra' nodes are padded with the zero embeddings
            # position one-hot
            pos_one_hot[i,:,:] = self._one_hot(ptype,self.max_n)
            # node i adj
            adj_i = torch.FloatTensor(g.get_adjacency().data).to(self._get_device())
            adj[i,:num_node,:num_node] = adj_i
            # src mask
            src_mask[head_count:head_count+self.nhead,:num_node,:num_node] = torch.stack([self._mask_generate(adj_i,num_node)[1:,1:]]*self.nhead,0)
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
            head_count += self.nhead
        return node_feature, pos_one_hot, adj, src_mask.to(torch.bool), tgt_mask.to(torch.bool).transpose(1,2), mem_mask.to(torch.bool), graph_sizes, true_types

    def encode(self, glist):
        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        bsize = len(graph_sizes) 

        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        # here we set the source sequence and the tgt sequence for the teacher forcing
        src_inp = node_feat.transpose(0,1) # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)

        
        #memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp,mask=tgt_mask)
        memory = memory.transpose(0,1).reshape(-1,self.max_n*self.nhid) # shape ( bsize, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.

        return self.fc1(memory), self.fc2(memory)

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
        memory = self.fc3(z).reshape(-1,self.max_n,self.nhid).transpose(0,1)

        G = [igraph.Graph(directed=True) for _ in range(bsize)]
        for g in G:
            g.add_vertex(type=self.START_SYMBOL)
            g.add_vertex(type=self.START_TYPE)
            g.add_edge(0,1)
            g.vs[0]['position'] = 0
            g.vs[1]['position'] = 1

        #memory = self.encoder(src_inp,mask=src_mask)

        finished = [False] * bsize
        for idx in range(2, self.max_n): # the first two type of nodes are certain
            node_one_hot, pos_one_hot, adj, _, tgt_mask, mem_mask, _, _ = self._prepare_features(G,self.max_n-1)
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
        memory = self.fc3(z).reshape(-1,self.max_n,self.nhid).transpose(0,1)

        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
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