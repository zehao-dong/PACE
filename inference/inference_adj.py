import os
import sys
import torch
import numpy as np

def inference():
    trainSet = torch.load('data/nasbench101/nasbench101_oo_testSet_split1.pt')
    validSet = torch.load('data/nasbench101/nasbench101_oo_testSet_split2.pt')
    k_v_map = {0: -1, 1: -0.5, 2: 0, 3: 0.5, 4: 1}
    features = []
    valid_accs = []
    test_accs = []
    times = []
    for dataset in [trainSet, validSet]:
        for i in range(len(dataset)):
            feature = []
            for j in range(len(dataset[i]['adj'])):
                for k in range(j, len(dataset[i]['adj'][0])):
                    feature.append(dataset[i]['adj'][j][k])
            op = []
            for x in dataset[i]['ops']:
                op.append(k_v_map[np.argmax(x)])
            feature.extend(op)
            print(feature)
            features.append(feature)
            valid_accs.append(dataset[i]['validation_acc'])
            test_accs.append(dataset[i]['test_acc'])
            times.append(dataset[i]['training_time'])

    features = torch.tensor(features)
    print(features.shape)
    pretrained_embeddings = {'embeddings': features, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
    torch.save(pretrained_embeddings, 'adj_oo_nasbench101' + '.pt')

if __name__ == '__main__':
    inference()