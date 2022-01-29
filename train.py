from transformers import RobertaTokenizer, BertTokenizer
from config import *
from trainer import Trainer
from trainer.utils import SimilarityCriterion
from trainer.scheduler import get_cosine_schedule_with_warmup
from dataloader import SimilarityData, Collator
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
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

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
named_param = model.named_parameters()
optimizer_grouped_parameters = [
    {'params': [p for n, p in named_param if not any(nd in n for nd in no_decay)], 'weight_decay': WD},
    {'params': [p for n, p in named_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

train_steps = 5750
warmup_steps = train_steps 
num_training_steps = train_steps * EPOCHS

optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
criterion = SimilarityCriterion(ctype=CRITERION_TYPE)

trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, use_wandb=USE_WANDB, scheduler=scheduler)

if LOAD_CKPT:
    trainer.load('ckpts/best.pt')

trainer.train(EPOCHS, print_freq=PRINT_FREQ)

