import torch
import torch.nn as nn
import math

class CosineSimilarityCriterion(nn.Module):
    def __init__(self, scale=[-1, 1]):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()
        self.scale = scale

    def forward(self, logits, labels):
        return self.get_loss(logits, labels)

    def get_loss(self, logits, labels):
        similarity = self.get_target(logits)
        similarity = self.scaling(similarity)
        loss = self.mse(similarity, labels)
        return loss 

    def scaling(self, x):
        floor, ceil = self.scale[0], self.scale[1]
        gap = ceil - floor
        x = 0.5 * (x + 1)
        x = gap * x + floor
        return x

    def get_target(self, logits):
        bs, dim = logits.shape[0], logits.shape[-1]
        logits = torch.reshape(logits, (bs//2, 2, dim))
        similarity = self.cos_sim(logits[:, 0, :], logits[:, 1, :])
        return similarity

class ArcScoreCriterion(nn.Module):
    def __init__(self, ctype='arc', window_type='identity', pow=1):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()
        self.ctype = ctype
        self.window_type = window_type
        self.pow = pow
    
    def forward(self, logits, labels):
        return self.get_loss(logits, labels)
    
    def get_loss(self, logits, labels):
        sim_score = self.get_target(logits)
        if self.ctype == 'square_arc':
            loss = self.mse(torch.square(sim_score), torch.square(labels))
        else:
            loss = self.mse(sim_score, labels)
        return loss

    def windowing(self, x):
        if self.window_type == 'identity':
            return x
        elif self.window_type == 'pow':
            return torch.pow(x, self.pow)
        elif self.window_type == 'original':
            return torch.cos(math.pi * (1 - x))
        else:
            print(f'{self.window_type} not supported!')

    def get_target(self, logits):
        bs, dim = logits.shape[0], logits.shape[-1]
        logits = torch.reshape(logits, (bs//2, 2, dim))
        similarity = self.cos_sim(logits[:, 0, :], logits[:, 1, :])
        sim_score = 1 - torch.acos(similarity) / math.pi
        target = self.windowing(sim_score) 
        return target 
