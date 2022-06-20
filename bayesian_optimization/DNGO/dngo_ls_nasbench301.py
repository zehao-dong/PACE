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
    genotypes = data['genotypes']
    pred_accs = data['predicted_accs']
    pred_times = data['predicted_runtimes']
    labels = torch.Tensor(pred_accs)
    labels = labels / 100.
    training_time = torch.Tensor(pred_times)
    print('loading finished. pretrained embeddings shape {}'.format(features.shape))
    return features, genotypes, labels, training_time

def get_samples(features, genotypes, labels, training_time, visited):
    init_inds = np.random.permutation(list(range(features.shape[0])))[:args.init_size]
    init_inds = torch.Tensor(init_inds).long()
    init_feat_samples = features[init_inds]
    init_geno_samples = [genotypes[i.item()] for i in init_inds]
    init_label_samples = labels[init_inds]
    init_time_samples = training_time[init_inds]
    for idx in init_inds:
        visited[idx] = True
    return init_feat_samples, init_geno_samples, init_label_samples, init_time_samples, visited

def propose_location(ei, features, genotypes, labels, training_time, visited):
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
    proposed_x, proposed_y, proposed_time = features[ind_dedup], labels[ind_dedup], training_time[ind_dedup]
    proposed_geno_samples = [genotypes[i.item()] for i in ind_dedup]
    return proposed_x, proposed_geno_samples, proposed_y, proposed_time, visited

def step(query, features, genotypes, labels, training_time, visited):
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
    return features[min_idx[i].item()], genotypes[min_idx[i].item()], labels[min_idx[i].item()], training_time[min_idx[i].item()], visited

def computation_aware_search(label_next, feat_samples, label_samples,
                             visited, best_trace, counter, rt, topk, features, genotypes,
                             labels, training_time,
                             CURR_BEST, CURR_BEST_GENOTYPE, MAX_BUDGET):
    indices = torch.argsort(label_samples.view(-1))
    for ind in indices[-topk:]:
        if label_samples[ind] not in label_next:
            feat_nn, geno_nn, label_nn, training_time_nn, visited = step(feat_samples[ind], features, genotypes, labels, training_time, visited)
            if label_nn.item() > CURR_BEST:
                print('FIND BEST VALID FROM NNS')
                CURR_BEST = label_nn.item()
                CURR_BEST_GENOTYPE = geno_nn
            feat_samples = torch.cat((feat_samples, feat_nn.view(1, -1)), dim=0)
            label_samples = torch.cat((label_samples.view(-1, 1), label_nn.view(1, 1)), dim=0)
            counter += 1
            rt += training_time_nn.item()
            best_trace['acc'].append(float(CURR_BEST))
            best_trace['genotype'].append(str(CURR_BEST_GENOTYPE))
            best_trace['counter'].append(counter)
            best_trace['time'].append(rt)
            if rt >= MAX_BUDGET:
                break

    return feat_samples, label_samples, visited, best_trace, rt, counter, CURR_BEST, CURR_BEST_GENOTYPE

def expected_improvement_search():
    """ implementation of CATE-DNGO-LS on the NAS-Bench-301 search space """
    CURR_BEST = 0.
    CURR_BEST_GENOTYPE = None
    PREV_BEST = 0
    MAX_BUDGET = 100
    window_size = 1024
    counter = 0
    round = 0
    rt = 0.
    visited = {}
    best_trace = defaultdict(list)
    features, genotypes, labels, training_time = load(args.embedding_path)
    feat_samples, geno_samples, label_samples, time_samples, visited = get_samples(features, genotypes, labels, training_time, visited)

    for feat, geno, acc, t in zip(feat_samples, geno_samples, label_samples, time_samples):
        counter += 1
        rt += t.item()
        if acc > CURR_BEST:
            CURR_BEST = acc
            CURR_BEST_GENOTYPE = geno
        best_trace['acc'].append(float(CURR_BEST))
        best_trace['genotype'].append(str(CURR_BEST_GENOTYPE))
        best_trace['counter'].append(counter)
        best_trace['time'].append(rt)

    while counter <= MAX_BUDGET:
        if round == args.rounds:
            feat_samples, geno_samples, label_samples, time_samples, visited = get_samples(features, genotypes, labels, training_time, visited)
            for feat, geno, acc, t in zip(feat_samples, geno_samples, label_samples, time_samples):
                counter += 1
                rt += t.item()
                if acc > CURR_BEST:
                    CURR_BEST = acc
                    CURR_BEST_GENOTYPE = geno
                best_trace['acc'].append(float(CURR_BEST))
                best_trace['genotype'].append(str(CURR_BEST_GENOTYPE))
                best_trace['counter'].append(counter)
                best_trace['time'].append(rt)
            round = 0
        print("current best: {}".format(CURR_BEST))
        print("current best genotype: {}".format(CURR_BEST_GENOTYPE))
        print("counter: {}".format(counter))
        print("rt: {}".format(rt))
        print(feat_samples.shape)
        print(label_samples.shape)
        model = DNGO(num_epochs=args.epochs, n_units_1=128, n_units_2=128, n_units_3=128, do_mcmc=False, normalize_output=False)
        model.train(X=feat_samples.numpy(), y=label_samples.view(-1).numpy(), do_optimize=True)
        print(model.network)
        m = []
        v = []
        chunks = int(features.shape[0] / window_size)
        if features.shape[0] % window_size > 0:
            chunks += 1
        features_split = torch.split(features, window_size, dim=0)
        for i in range(chunks):
            m_split, v_split = model.predict(features_split[i].numpy())
            m.extend(list(m_split))
            v.extend(list(v_split))

        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        u = (mean - torch.Tensor([1.0]).expand_as(mean)) / sigma
        ei = sigma * (u * stats.norm.cdf(u) + 1 + stats.norm.pdf(u))
        feat_next, geno_next, label_next, time_next, visited = \
            propose_location(ei, features, genotypes, labels, training_time, visited)
        print('proposed top-k: {}'.format(label_next))

        # add proposed networks to the pool
        for feat, geno, acc, t in zip(feat_next, geno_next, label_next, time_next):
            if acc > CURR_BEST:
                print('FIND BEST VALID FROM DNGO')
                CURR_BEST = acc
                CURR_BEST_GENOTYPE = geno
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            label_samples = torch.cat((label_samples.view(-1, 1), acc.view(1, 1)), dim=0)
            counter += 1
            rt += t.item()
            best_trace['acc'].append(float(CURR_BEST))
            best_trace['genotype'].append(str(CURR_BEST_GENOTYPE))
            best_trace['counter'].append(counter)
            best_trace['time'].append(rt)
            if counter >= MAX_BUDGET:
                break
        if args.computation_aware_search:
            feat_samples, label_samples, visited, best_trace, rt, counter, CURR_BEST, CURR_BEST_GENOTYPE = \
                computation_aware_search(label_next, feat_samples, label_samples,
                                         visited, best_trace, counter, rt, args.topk, features,
                                         genotypes, labels, training_time,
                                         CURR_BEST, CURR_BEST_GENOTYPE, MAX_BUDGET)
        if PREV_BEST < CURR_BEST:
            PREV_BEST = CURR_BEST
        else:
            round += 1

    res = dict()
    res['acc'] = best_trace['acc']
    res['genotype'] = best_trace['genotype']
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
    parser = argparse.ArgumentParser(description="DNGO")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--epochs', type=int, default=30, help='outer loop epochs')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--topk', type=int, default=5, help='acquisition samples')
    parser.add_argument('--rounds', type=int, default=20, help='rounds allowed for local minimum')
    parser.add_argument('--output_path', type=str, default='bo', help='bo')
    parser.add_argument('--embedding_path', type=str, default='cate_nasbench301.pt')
    parser.add_argument('--dataset', type=str, default='nasbench301')
    parser.add_argument('--computation_aware_search', type=bool, default=True)
    args = parser.parse_args()
    expected_improvement_search()