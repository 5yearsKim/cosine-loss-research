import torch
from torch import nn


class Pooling(nn.Module):
    def __init__(self, pooling_type='mean'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, hidden_state, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1)
        masked_hidden = expanded_mask * hidden_state
        eps = 1e-5
        attention_len = torch.sum(attention_mask, dim=1) + eps
        mean_pooled = torch.div(masked_hidden.sum(dim=1), attention_len.unsqueeze(-1))
        return mean_pooled