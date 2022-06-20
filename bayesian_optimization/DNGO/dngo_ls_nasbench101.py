import os
import sys
sys.path.insert(0, os.getcwd())
from pybnn.dngo import DNGO
import argparse
import json
import torch
import scipy.stats as stats
import numpy as np
from collections import defaultdict

def load(path):
    data = torch.load(path, map_location=torch.device('cpu'))
    print('load pretrained embeddings from {}'.format(path))
    features = data['embeddings']
    valid_labels = data['valid_accs']
    test_labels = data['test_accs']
    training_time = data['times']
    test_labels = torch.Tensor(test_labels)
    valid_labels = torch.Tensor(valid_labels)
    training_time = torch.Tensor(training_time)
    print('loading finished. pretrained embeddings shape {}'.format(features.shape))
    return features, valid_labels, test_labels, training_time

def get_samples(features, valid_labels, test_labels, training_time, visited):
    init_inds = np.random.permutation(list(range(features.shape[0])))[:args.init_size]
    ind_dedup = []
    for idx in init_inds:
        if idx not in visited:
            visited[idx] = True
            ind_dedup.append(idx)
    init_inds = torch.Tensor(ind_dedup).long()
    init_feat_samples = features[init_inds]
    init_valid_label_samples = valid_labels[init_inds]
    init_test_label_samples = test_labels[init_inds]
    init_time_samples = training_time[init_inds]
    return init_feat_samples, init_valid_label_samples, init_test_label_samples, init_time_samples, visited

def propose_location(ei, features, valid_labels, test_labels, training_time, visited):
    k = args.topk
    ei = ei.view(-1)
    print('remaining length of indices set:', len(features) - len(visited))
    indices = torch.argsort(ei)[-k:]
    ind_dedup = []
    for idx in indices:
        if idx not in visited:
            visited[idx] = True
            ind_dedup.append(idx)
    ind_dedup = torch.Tensor(ind_dedup).long()
    proposed_x, proposed_y_valid, proposed_y_test, proposed_time = features[ind_dedup], valid_labels[ind_dedup], test_labels[ind_dedup], training_time[ind_dedup]
    return proposed_x, proposed_y_valid, proposed_y_test, proposed_time, visited

def step(query, features, valid_labels, test_labels, training_time, visited):
    dist = torch.norm(features - query.view(1, -1), dim=1)
    knn = (-1 * dist).topk(dist.shape[0])
    min_dist, min_idx = knn.values, knn.indices
    i = 0
    while True:
        if len(visited) == dist.shape[0]:
            print("cannot find in the dataset")
            exit()
        if min_idx[i].item() not in visited:
            visited[min_idx[i].item()] = True
            break
        i += 1

    return features[min_idx[i].item()], valid_labels[min_idx[i].item()], test_labels[min_idx[i].item()], training_time[min_idx[i].item()], visited

def computation_aware_search(label_next_valid, feat_samples, valid_label_samples, test_label_samples,
                             visited, best_trace, counter, rt, topk, features,
                             valid_labels, test_labels, training_time,
                             BEST_VALID_ACC, BEST_TEST_ACC, CURR_BEST_VALID,
                             CURR_BEST_TEST, MAX_BUDGET):
    indices = torch.argsort(valid_label_samples.view(-1))
    for ind in indices[-topk:]:
        if valid_label_samples[ind] not in label_next_valid:
            feat_nn, valid_label_nn, test_label_nn, training_time_nn, visited = \
                step(feat_samples[ind], features, valid_labels, test_labels, training_time, visited)
            if valid_label_nn.item() > CURR_BEST_VALID:
                CURR_BEST_VALID = valid_label_nn.item()
                CURR_BEST_TEST = test_label_nn.item()
            feat_samples = torch.cat((feat_samples, feat_nn.view(1, -1)), dim=0)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), valid_label_nn.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), test_label_nn.view(1, 1)), dim=0)
            counter += 1
            rt += training_time_nn.item()
            best_trace['regret_validation'].append(float(BEST_VALID_ACC - CURR_BEST_VALID))
            best_trace['regret_test'].append(float(BEST_TEST_ACC - CURR_BEST_TEST))
            best_trace['time'].append(rt)
            best_trace['counter'].append(counter)
            if counter >= MAX_BUDGET:
                break

    return feat_samples, valid_label_samples, test_label_samples, visited, \
           best_trace, rt, counter, CURR_BEST_VALID, CURR_BEST_TEST


def expected_improvement_search():
    """ implementation of CATE-DNGO-LS on the NAS-Bench-101 search space """
    BEST_TEST_ACC = 0.943175752957662 
    BEST_VALID_ACC = 0.9505542516708374
    PREV_BEST = 0
    CURR_BEST_VALID = 0.
    CURR_BEST_TEST = 0.
    MAX_BUDGET = 150
    window_size = 1024
    counter = 0
    round = 0
    rt = 0.
    visited = {}
    best_trace = defaultdict(list)
    features, valid_labels, test_labels, training_time = load(args.embedding_path)
    feat_samples, valid_label_samples, test_label_samples, time_samples, visited = get_samples(features, valid_labels, test_labels, training_time, visited)

    for feat, acc_valid, acc_test, t in zip(feat_samples, valid_label_samples, test_label_samples, time_samples):
        counter += 1
        rt += t.item()
        if acc_valid > CURR_BEST_VALID:
            CURR_BEST_VALID = acc_valid
            CURR_BEST_TEST = acc_test
        best_trace['regret_validation'].append(float(BEST_VALID_ACC - CURR_BEST_VALID))
        best_trace['regret_test'].append(float(BEST_TEST_ACC - CURR_BEST_TEST))
        best_trace['time'].append(rt)
        best_trace['counter'].append(counter)

    while counter <= MAX_BUDGET:
        if round == args.rounds:
            feat_samples, valid_label_samples, test_label_samples, time_samples, visited = \
                get_samples(features, valid_labels, test_labels, training_time, visited)
            for feat, acc_valid, acc_test, t in zip(feat_samples, valid_label_samples, test_label_samples, time_samples):
                counter += 1
                rt += t.item()
                if acc_valid > CURR_BEST_VALID:
                    CURR_BEST_VALID = acc_valid
                    CURR_BEST_TEST = acc_test
                best_trace['regret_validation'].append(float(BEST_VALID_ACC - CURR_BEST_VALID))
                best_trace['regret_test'].append(float(BEST_TEST_ACC - CURR_BEST_TEST))
                best_trace['time'].append(rt)
                best_trace['counter'].append(counter)
            round = 0

        print("current best validation: {}".format(CURR_BEST_VALID))
        print("current best test: {}".format(CURR_BEST_TEST))
        print("counter: {}".format(counter))
        print("rt: {}".format(rt))
        print(feat_samples.shape)
        print(valid_label_samples.shape)
        print('begin training BO model')
        #model = DNGO(num_epochs=args.epochs, n_units=128, do_mcmc=False, normalize_output=False)
        model = DNGO(num_epochs=args.epochs, n_units_1=128, n_units_2=128, n_units_3=128, do_mcmc=False, normalize_output=False)
        model.train(X=feat_samples.numpy(), y=valid_label_samples.view(-1).numpy(), do_optimize=True)
        print('BO model training finished')
        print(model.network)
        m = []
        v = []
        chunks = int(features.shape[0] / window_size)
        if features.shape[0] % window_size > 0:
            chunks += 1
        features_split = torch.split(features, window_size, dim=0)
        for i in range(chunks):
            if i % 100 == 0:
                print('processing chunk: {}'.format(i))
            m_split, v_split = model.predict(features_split[i].numpy())
            m.extend(list(m_split))
            v.extend(list(v_split))

        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        u = (mean - torch.Tensor([1.0]).expand_as(mean)) / sigma
        ei = sigma * (u * stats.norm.cdf(u) + 1 + stats.norm.pdf(u))
        print('begin determining next position')
        feat_next, label_next_valid, label_next_test, time_next, visited = \
            propose_location(ei, features, valid_labels, test_labels, training_time, visited)
        print('next positions detected')
        # add proposed networks to the pool
        for feat, acc_valid, acc_test, t in zip(feat_next, label_next_valid, label_next_test, time_next):
            if acc_valid > CURR_BEST_VALID:
                CURR_BEST_VALID = acc_valid
                CURR_BEST_TEST = acc_test
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), acc_valid.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), acc_test.view(1, 1)), dim=0)
            counter += 1
            rt += t.item()
            best_trace['regret_validation'].append(float(BEST_VALID_ACC - CURR_BEST_VALID))
            best_trace['regret_test'].append(float(BEST_TEST_ACC - CURR_BEST_TEST))
            best_trace['time'].append(rt)
            best_trace['counter'].append(counter)
            if counter >= MAX_BUDGET:
                break

        if args.computation_aware_search:
            print('begin computation aware searching')
            feat_samples, valid_label_samples, test_label_samples, \
            visited, best_trace, rt, counter, CURR_BEST_VALID, CURR_BEST_TEST =\
                computation_aware_search(label_next_valid, feat_samples, valid_label_samples, test_label_samples,
                                         visited, best_trace, counter, rt, args.topk, features,
                                         valid_labels, test_labels, training_time,
                                         BEST_VALID_ACC, BEST_TEST_ACC, CURR_BEST_VALID, CURR_BEST_TEST,
                                         MAX_BUDGET)
        if PREV_BEST < CURR_BEST_VALID:
            PREV_BEST = CURR_BEST_VALID
        else:
            round += 1

    res = dict()
    res['regret_validation'] = best_trace['regret_validation']
    res['regret_test'] = best_trace['regret_test']
    res['runtime'] = best_trace['time']
    res['counter'] = best_trace['counter']
    save_path = args.dataset + '/' + args.output_path + '/' + 'dim{}'.format(args.dim)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print('save to {}'.format(save_path))
    fh = open(os.path.join(save_path, 'run_{}.json'.format(args.seed)),'w')
    json.dump(res, fh)
    fh.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CATE-DNGO-LS")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--epochs', type=int, default=30, help='outer loop epochs')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--topk', type=int, default=5, help='acquisition samples')
    parser.add_argument('--rounds', type=int, default=20, help='rounds allowed for local minimum')
    parser.add_argument('--output_path', type=str, default='bo', help='bo')
    parser.add_argument('--embedding_path', type=str, default='cate_nasbench101.pt')
    parser.add_argument('--dataset', type=str, default='nasbench101')
    parser.add_argument('--computation_aware_search', type=bool, default=True)
    args = parser.parse_args()
    expected_improvement_search()