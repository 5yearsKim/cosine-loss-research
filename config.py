
# ['squre_arc', 'arc', 'cos_sim', 'cos_no_scale']
CRITERION_TYPE = 'cos_no_scale'
MODEL_TYPE = 'roberta'

if MODEL_TYPE == 'roberta':
    TKNZR_PATH = './tknzr/roberta_tknzr'
elif MODEL_TYPE == 'bert':
    TKNZR_PATH = './tknzr/bert_tknzr'


LOAD_CKPT = False 
USE_WANDB = False 

LR = 1e-5
WD = 0.01
BS = 22
EPOCHS =  5
PRINT_FREQ = 50
