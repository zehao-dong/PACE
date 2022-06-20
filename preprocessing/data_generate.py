import os
import numpy as np
import torch
import random
import argparse
from tqdm import tqdm

def sample_from_data(data, K=2, maxDist=2e6, metric=None):

    sorted_data = sorted(data.values(), key=lambda x: x[metric])

    cnt = 0
    data_pair = {}

    head = 0
    pbar = tqdm(sorted_data)
    #print(len(sorted_data))
    #for i, data_i in enumerate(sorted_data):
    for i, data_i in enumerate(pbar):
        #if i % 10000 == 0:
        #    print(i)
        while (head < i) and (sorted_data[head][metric] + maxDist < data_i[metric]):
            head += 1
        picks = list(range(head, i))
        random.shuffle(picks)
        if len(picks) > K:
            picks = picks[:K]
        for j in picks:
            data_pair[cnt] = (sorted_data[i]['index'], sorted_data[j]['index'])
            cnt += 1

    return data_pair

def argLoader():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nasbench101', help='nasbench101/nasbench301/oo')
    parser.add_argument('--flag', type=str, default='build_pair', help='extract_seq/build_pair')
    parser.add_argument('--data_path', type=str, default='data/nasbench101/', help='input/output path')
    parser.add_argument('--k', type=int, default=2, help='number of architecture pairs')
    parser.add_argument('--d', type=int, default=2e6, help='computation constraint')
    parser.add_argument('--metric', type=str, default='params', help='params/flops')
    parser.add_argument('--n_percent', type=float, default=0.95, help='train/val split')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import json
    args = argLoader()
    flag = args.flag
    torch.set_num_threads(1)
    if args.flag == 'extract_seq':
        train_data = {}
        test_data = {}
        if args.dataset == 'nasbench101':
            with open('data/nasbench101.json') as f:
                archs = json.load(f)
            for i in range(int(len(archs) * args.n_percent)):
                train_data[i] = {
                    'index': i,
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'params': archs[str(i)]['parameters'],
                    'validation_accuracy': archs[str(i)]['validation_accuracy'],
                    'test_accuracy': archs[str(i)]['test_accuracy'],
                    'training_time': archs[str(i)]['training_time']
                }
            for i in range(int(len(archs) * args.n_percent), len(archs)):
                test_data[i - int(len(archs) * args.n_percent)] = {
                    'index': i - int(len(archs) * args.n_percent),
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'params': archs[str(i)]['parameters'],
                    'validation_accuracy': archs[str(i)]['validation_accuracy'],
                    'test_accuracy': archs[str(i)]['test_accuracy'],
                    'training_time': archs[str(i)]['training_time']
                }
        elif args.dataset == 'nasbench301':
            with open('data/nasbench301_proxy.json') as f:
                archs = json.load(f)
            for i in range(int(len(archs) * args.n_percent)):
                train_data[i] = {
                    'index': i,
                    'adj': archs[str(i)]['adjacency_matrix_nas101_format'],
                    'ops': archs[str(i)]['operations_nas101_format'],
                    'genotype': archs[str(i)]['genotype'],
                    'params': archs[str(i)]['params'],
                    'flops': archs[str(i)]['flops'],
                    'predicted_acc': archs[str(i)]['predicted_acc'],
                    'predicted_runtime': archs[str(i)]['predicted_runtime']
                    }
            for i in range(int(len(archs) * args.n_percent), len(archs)):
                test_data[i - int(len(archs) * args.n_percent)] = {
                    'index': i - int(len(archs) * args.n_percent),
                    'adj': archs[str(i)]['adjacency_matrix_nas101_format'],
                    'ops': archs[str(i)]['operations_nas101_format'],
                    'genotype': archs[str(i)]['genotype'],
                    'params': archs[str(i)]['params'],
                    'flops': archs[str(i)]['flops'],
                    'predicted_acc': archs[str(i)]['predicted_acc'],
                    'predicted_runtime': archs[str(i)]['predicted_runtime']
                }
        save_dir = os.path.join('data/', args.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(train_data, os.path.join(save_dir, 'train_data.pt'))
        torch.save(test_data, os.path.join(save_dir, 'test_data.pt'))
    elif args.flag == 'build_pair':
        save_dir = os.path.join('data/', args.dataset)
        train_data = torch.load(os.path.join(save_dir, 'train_data.pt'))
        test_data = torch.load(os.path.join(save_dir, 'test_data.pt'))
        train_data_pair = sample_from_data(train_data, K=args.k, maxDist=args.d, metric=args.metric)
        test_data_pair = sample_from_data(test_data, K=args.k, maxDist=args.d, metric=args.metric)
        train_name = 'train_pair_k{}_d{}_metric_{}.pt'.format(args.k, args.d, args.metric)
        test_name = 'test_pair_k{}_d{}_metric_{}.pt'.format(args.k, args.d, args.metric)
        torch.save(train_data_pair, os.path.join(save_dir, train_name))
        torch.save(test_data_pair, os.path.join(save_dir, test_name))