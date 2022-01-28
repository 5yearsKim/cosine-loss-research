from transformers import RobertaTokenizer, BertTokenizer, AdamW
from config import *
from trainer import Trainer
from trainer.utils import SimilarityCriterion
from dataloader import SimilarityData, Collator
from torch.utils.data import DataLoader
import torch
from models import SentenceRoberta, SentenceBert

import wandb

if USE_WANDB:
    wandb.init(project="cosine-research", entity="akaai", name=f'{MODEL_TYPE} {CRITERION_TYPE} square bs={BS},lr={LR}')
    wandb.config = {
      "learning_rate": LR,
      "batch_size": BS,
    }

model_type = 'roberta'


train_from = ['./data/glue/STS-B/train.tsv']
val_from = ['./data/glue/STS-B/dev.tsv']

train_set = SimilarityData(file_from=train_from)
val_set = SimilarityData(file_from=val_from)

if MODEL_TYPE == 'roberta':
    tknzr = RobertaTokenizer.from_pretrained(TKNZR_PATH)
    model = SentenceRoberta()
elif MODEL_TYPE == 'bert':
    tknzr = BertTokenizer.from_pretrained(TKNZR_PATH)
    model = SentenceBert()

# tknzr = BertTokenizer.from_pretrained(TKNZR_PATH)
collator = Collator(tknzr=tknzr)

train_loader = DataLoader(train_set, batch_size=BS, collate_fn=collator)
val_loader = DataLoader(val_set, batch_size=BS, collate_fn=collator)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
criterion = SimilarityCriterion(ctype=CRITERION_TYPE)

trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, use_wandb=USE_WANDB)


if LOAD_CKPT:
    trainer.load('ckpts/best.pt')

trainer.train(EPOCHS, print_freq=PRINT_FREQ)

