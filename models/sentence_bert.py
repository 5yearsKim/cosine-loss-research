import torch.nn as nn
from transformers import BertModel

class SentenceBert(nn.Module):
    def __init__(self, pretrained_path='bert-small', out_dims=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.out_dims = out_dims 
        self.cls = nn.Linear(768, self.out_dims)

    def forward(self, inputs):
        bout = self.bert(**inputs)
        vector = self.cls(bout.pooler_output)
        return vector