
MODEL_TYPE = 'roberta'

if MODEL_TYPE == 'roberta':
    TKNZR_PATH = './tknzr/roberta_tknzr'
elif MODEL_TYPE == 'bert':
    TKNZR_PATH = './tknzr/bert_tknzr'


LOAD_CKPT = False 
USE_WANDB = True 

LR = 2e-5
WD = 0
BS = 8 
EPOCHS =  10 
PRINT_FREQ = 1 
