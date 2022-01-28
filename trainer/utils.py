import torch
import torch.nn as nn
import math

class AverageMeter:
    """code from TNT"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_accuracy(y_true, y_prob, threshold=0.5):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > threshold
    y_true = y_true > threshold 
    return (y_true == y_prob).sum().item() / y_true.size(0)


class SimilarityCriterion:
    def __init__(self, ctype='normal'):
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()
        self.ctype = ctype

    def __call__(self, logits, labels):
        if self.ctype == 'arc':
            return self.get_arc_loss(logits, labels)
        elif self.ctype == 'square_arc':
            return self.get_square_arc_loss(logits, labels)
        elif self.ctype == 'normal':
            return self.get_normal_loss(logits, labels)
        else:
            print(f'{self.ctype} not supported!')

    def get_normal_loss(self, logits, labels):
        bs, dim = logits.shape[0], logits.shape[-1]
        logits = torch.reshape(logits, (bs//2, 2, dim))
        similarity = self.cos_sim(logits[:, 0, :], logits[:, 1, :])
        similarity = 0.5 * (similarity + 1) 
        loss = self.mse(similarity, labels)
        return loss 
    
    def get_arc_loss(self, logits, labels):
        sim_score = self.get_sim_score(logits)
        loss = self.mse(sim_score, labels)
        return loss

    def get_square_arc_loss(self, logits, labels):
        sim_score = self.get_sim_score(logits)
        loss = self.mse(torch.square(sim_score), torch.square(labels))
        return loss


    def get_sim_score(self, logits):
        bs, dim = logits.shape[0], logits.shape[-1]
        logits = torch.reshape(logits, (bs//2, 2, dim))
        similarity = self.cos_sim(logits[:, 0, :], logits[:, 1, :])
        sim_score = 1 - torch.acos(similarity) / math.pi
        return sim_score

    def get_acc(self, logits, labels):
        sim_score = self.get_sim_score(logits)
        acc = get_accuracy(labels, sim_score, threshold=0.7)
        return acc


def pearson_r(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    def _get_ranks(x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(len(x))
        return ranks
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)