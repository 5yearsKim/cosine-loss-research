
# ['squre_arc', 'arc', 'normal']
CRITERION_TYPE = 'square_arc'
MODEL_TYPE = 'roberta'

if MODEL_TYPE == 'roberta':
    TKNZR_PATH = './tknzr/roberta_tknzr'
elif MODEL_TYPE == 'bert':
    TKNZR_PATH = './tknzr/bert_tknzr'


LOAD_CKPT = False 
USE_WANDB = True 

LR = 2e-5
WD = 0
BS = 22 
EPOCHS =  10 
PRINT_FREQ = 50 
