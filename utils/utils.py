import random
import numpy as np
import torch
import torch.nn.functional as F

def floyed(r):
    """
    :param r: a numpy NxN matrix with float 0,1
    :return: a numpy NxN matrix with float 0,1
    """
    r = np.array(r)
    N = r.shape[0]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if r[i, k] > 0 and r[k, j] > 0:
                    r[i, j] = 1
    return r

def prepare_graph(graph, config):
    Xs, Rs, Ls = zip(*graph)
    ls = [len(it) for it in Xs]
    maxL = max(ls)
    inputs = []
    masks = []
    labels = []
    adjs = []
    for x, r, L, l in zip(Xs, Rs, Ls, ls):
        input_i = torch.LongTensor(x)
        label_i = torch.LongTensor(L)
        mask_i = torch.from_numpy(floyed(r)).float()
        adj_i = np.diag(np.ones(maxL))
        r_np = np.array(r)
        num_node =r_np.shape[0] 
        adj_i[:num_node,:num_node] = r_np 
        adj_i = torch.from_numpy(adj_i).float()
        #mask_i = torch.from_numpy(np.asarray(r)).float() # no floyed
        padded_input_i = F.pad(input_i, (0, maxL - l), "constant", config.PAD)
        padded_label_i = F.pad(label_i, (0, maxL - l), "constant", config.PAD)
        padded_mask_i = F.pad(mask_i, (0, maxL - mask_i.shape[1], 0, maxL - mask_i.shape[1]), "constant", config.PAD)
        inputs.append(padded_input_i)
        masks.append(padded_mask_i)
        labels.append(padded_label_i)
        adjs.append(adj_i)
    return torch.stack(inputs), torch.stack(masks), torch.stack(labels), torch.stack(adjs)

def createCrossMask(n, m, N, M):
    mask = torch.zeros(N+M, N+M)
    mask[:n, :n] = torch.ones(n, n)
    mask[N:N+m, N:N+m] = torch.ones(m, m)
    mask[:n, N:N+m] = torch.ones(n, m)
    mask[N:N+m, :n] = torch.ones(m, n)
    return mask

def prepareCrossAttention(g1, g2):
    Xs, _, _ = zip(*g1)
    Ys, _, _ = zip(*g2)
    lXs = [len(it) for it in Xs]
    lYs = [len(it) for it in Ys]
    maxLx = max(lXs)
    maxLy = max(lYs)
    masks = []
    for lx, ly in zip(lXs, lYs):
        mask = createCrossMask(lx, ly, maxLx, maxLy)
        masks.append(mask)
    return torch.stack(masks)

def sequence_corruption(seq, config):
    label = []
    masked_seq = []
    for it in seq:
        r = random.random()
        if r < config.corruption_rate:
            label.append(it)
            rr = random.random()
            if rr < 0.8:
                masked_seq.append(config.MASK)
            else:
                masked_seq.append(random.choice(list(range(config.n_vocab))))
        else:
            label.append(config.PAD)
            masked_seq.append(it)

    return masked_seq, label

def apply_mask(graph_pairs, config):
    g1s = []
    g2s = []
    for g1, g2 in graph_pairs:
        X1, R1 = g1
        X2, R2 = g2
        X1_, L1 = sequence_corruption(X1, config)
        X2_, L2 = sequence_corruption(X2, config)
        g1s.append([X1_, R1, L1])
        g2s.append([X2_, R2, L2])
    return g1s, g2s

def prepare_train(graph_pairs, config):
    # Applying Mask on Labels
    masked_g1, masked_g2 = apply_mask(graph_pairs, config)

    X, maskX, labelX, adjX = prepare_graph(masked_g1, config)
    maskX_ = maskX.transpose(-2, -1)
    Y, maskY, labelY, adjY = prepare_graph(masked_g2, config)
    maskY_ = maskY.transpose(-2, -1)
    maskXY = prepareCrossAttention(masked_g1, masked_g2)

    return X, maskX, maskX_, Y, maskY, maskY_, maskXY, torch.cat([labelX, labelY], dim=-1), adjX, adjY

def analytic():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,1)
    data_1 = torch.load('data/nasbench101/train_data.pt')
    data_2 = torch.load('data/nasbench101/test_data.pt')
    params = []
    for data in [data_1, data_2]:
        for i in range(len(data)):
            params.append(data[i]['params'])
    axes.hist(params, bins=50, label=['params'])
    axes.set_xlabel('number of trainable model parameters', fontsize=12)
    axes.set_ylabel('frequency', fontsize=12)
    axes.set_title('Histogram for model parameters on NASBench-101', fontsize=13)
    plt.show()

if __name__ == '__main__':
    analytic()