import torch
import torch.nn as nn
from torch.autograd import Variable

class LabelSmoothing(nn.Module):
    def __init__(self, config):
        super(LabelSmoothing, self).__init__()
        self.crit = nn.KLDivLoss(size_average=False)
        self.pad_idx = config.PAD
        self.confidence = 1.0 - config.label_smoothing
        self.smoothing = config.label_smoothing
        self.size = config.n_vocab

    def forward(self, predicts, target):
        assert self.size == predicts.size(1)
        dist = torch.full_like(predicts, self.smoothing / (self.size - 2))
        dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        dist[:, self.pad_idx] = 0
        mask_idx = torch.nonzero(target.data == self.pad_idx)
        if mask_idx.dim() > 0:
            dist.index_fill_(0, mask_idx.squeeze(), 0.0)
        return self.crit(predicts, Variable(dist, requires_grad=False))


class KLDivLoss(nn.Module):
    def __init__(self, config):
        super(KLDivLoss, self).__init__()
        self.crit = LabelSmoothing(config)

    def forward(self, predicts, target, norm=1.0):
        loss = self.crit(predicts.contiguous().view(-1,predicts.size(-1)), target.contiguous().view(-1))
        return loss / norm

