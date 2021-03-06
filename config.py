
# ['squre_arc', 'arc', 'cos_sim']
CRITERION_TYPE = 'arc'
MODEL_TYPE = 'roberta'

if MODEL_TYPE == 'roberta':
    TKNZR_PATH = './tknzr/roberta_tknzr'
elif MODEL_TYPE == 'bert':
    TKNZR_PATH = './tknzr/bert_tknzr'


LOAD_CKPT = False 
USE_WANDB = False 

LR = 2e-5
WD = 0.01
BS = 32
EPOCHS =  5
PRINT_FREQ = 50
