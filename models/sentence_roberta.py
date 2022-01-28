import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel

class SentenceRoberta(nn.Module):
    def __init__(self, pretrained_path='roberta-base', out_dims=256):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(pretrained_path)
        self.out_dims = out_dims 
        self.cls = nn.Linear(768, self.out_dims)

    def forward(self, inputs):
        bout = self.bert(**inputs)
        vector = self.cls(bout.pooler_output)
        return vector
