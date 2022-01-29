import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel
from .pooling import Pooling

class SentenceRoberta(nn.Module):
    def __init__(self, pretrained_path='roberta-base', out_dims=256):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(pretrained_path)
        self.out_dims = out_dims 
        # self.cls = nn.Linear(768, self.out_dims)
        self.cls = Pooling(pooling_type='mean')

    @property
    def embedding_dimension(self):
        return self.bert.config.hidden_size

    def forward(self, inputs):
        bout = self.bert(**inputs)
        vector = self.cls(bout.last_hidden_state, inputs.attention_mask)
        return vector
