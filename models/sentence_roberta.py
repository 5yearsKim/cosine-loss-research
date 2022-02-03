import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel
from .pooling import Pooling

class SentenceRoberta(nn.Module):
    def __init__(self, pretrained_path='roberta-base', cls_type='fc'):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(pretrained_path)
        self.out_dims = 128 
        if cls_type == 'fc':
            self.cls = nn.Linear(768, self.out_dims)
        elif cls_type == 'mean':
            self.cls = Pooling(pooling_type='mean')
        else:
            print(f'cls_type {cls_type} not supported!')
        self.cls_type = cls_type

    @property
    def embedding_dimension(self):
        return self.bert.config.hidden_size

    def forward(self, inputs):
        bout = self.bert(**inputs)
        if self.cls_type == 'mean':
            vector = self.cls(bout.last_hidden_state, inputs.attention_mask)
        else:
            vector = self.cls(bout.pooler_output)
        return vector
