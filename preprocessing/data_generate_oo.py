import numpy as np
import torch
import random
import argparse

def sample_from_data(data, K=2, maxDist=2e6, metric=None): #complexity from N**2 -> N

    sorted_data = sorted(data.values(), key=lambda x: x[metric]) #Nlog(N)

    cnt = 0
    data_pair = {}

    head = 0
    for i, data_i in enumerate(sorted_data):
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
    parser.add_argument('--flag', type=str, default='extract_seq', help='extract_seq | build_pair')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import json
    args = argLoader()
    torch.set_num_threads(1)
    if args.flag == 'extract_seq':
        for f_name in ['nasbench101_oo_train.json', 'nasbench101_oo_test.json']:
            with open('data/' + f_name) as f:
                    archs = json.load(f)
            train_data = {}
            test_data = {}
            for i in range(int(len(archs) * 0.95)):
                train_data[i] = {
                    'index': i,
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'validation_acc': archs[str(i)]['validation_accuracy'],
                    'params': archs[str(i)]['parameters'],
                    'test_acc': archs[str(i)]['test_accuracy'],
                    'training_time': archs[str(i)]['training_time'],
                    }
            for i in range(int(len(archs) * 0.95), len(archs)):
                test_data[i - int(len(archs) * 0.95)] = {
                    'index': i - int(len(archs) * 0.95),
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'validation_acc': archs[str(i)]['validation_accuracy'],
                    'params': archs[str(i)]['parameters'],
                    'test_acc': archs[str(i)]['test_accuracy'],
                    'training_time': archs[str(i)]['training_time'],
                }
            if f_name.split('.json')[0].split('_')[-1] == 'train':
                torch.save(train_data, 'data/nasbench101/nasbench101_oo_trainSet_train.pt')
                torch.save(test_data, 'data/nasbench101/nasbench101_oo_trainSet_validation.pt')
            if f_name.split('.json')[0].split('_')[-1] == 'test':
                torch.save(train_data, 'data/nasbench101/nasbench101_oo_testSet_split1.pt')
                torch.save(test_data, 'data/nasbench101/nasbench101_oo_testSet_split2.pt')
    elif args.flag == 'build_pair':
        train_data = torch.load('data/nasbench101/nasbench101_oo_trainSet_train.pt')
        test_data = torch.load('data/nasbench101/nasbench101_oo_trainSet_validation.pt')
        train_data_pair = sample_from_data(train_data, K=2, maxDist=2000000, metric='params')
        test_data_pair = sample_from_data(test_data, K=2, maxDist=2000000, metric='params')
        torch.save(train_data_pair, 'data/nasbench101/oo_train_pairs_k2_params_dist2e6.pt')
        torch.save(test_data_pair, 'data/nasbench101/oo_validation_pairs_k2_params_dist2e6.pt')
