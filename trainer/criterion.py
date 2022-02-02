import torch
import torch.nn as nn
import math

class CosineSimilarityCriterion(nn.Module):
    def __init__(self, ctype='cos_sim'):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()
        self.ctype = ctype

    def forward(self, logits, labels):
        return self.get_loss(logits, labels)

    def get_loss(self, logits, labels):
        similarity = self.get_target(logits)
        if self.ctype != 'cos_no_scale':
            similarity = 0.5 * (similarity + 1)
        loss = self.mse(similarity, labels)
        return loss 

    def get_target(self, logits):
        bs, dim = logits.shape[0], logits.shape[-1]
        logits = torch.reshape(logits, (bs//2, 2, dim))
        similarity = self.cos_sim(logits[:, 0, :], logits[:, 1, :])
        return similarity

class ArcScoreCriterion(nn.Module):
    def __init__(self, ctype='arc'):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()
        self.ctype = ctype
    
    def forward(self, logits, labels):
        return self.get_loss(logits, labels)
    
    def get_loss(self, logits, labels):
        sim_score = self.get_target(logits)
        if self.ctype == 'square_arc':
            loss = self.mse(torch.square(sim_score), torch.square(labels))
        else:
            loss = self.mse(sim_score, labels)
        return loss

    def get_target(self, logits):
        bs, dim = logits.shape[0], logits.shape[-1]
        logits = torch.reshape(logits, (bs//2, 2, dim))
        similarity = self.cos_sim(logits[:, 0, :], logits[:, 1, :])
        sim_score = 1 - torch.acos(similarity) / math.pi
        return sim_score
