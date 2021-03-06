from .utils import AverageMeter, pearson_r, spearman_correlation
import torch
import wandb

class Trainer:
    def __init__(self, model, optim, criterion, train_loader, val_loader, val_best_path='ckpts/best.pt', \
            use_wandb=False, scheduler=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optim = optim
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_best_path = val_best_path
        self.loss_meter = AverageMeter() 
        self.val_best = float('inf') 
        self.use_wandb = use_wandb
        self.scheduler = scheduler

        # self.validate(0)

    def train(self, epochs, print_freq=50):
        for epoch in range(epochs):
            print("\n")
            self.model.train()
            self.loss_meter.reset()
            for i, (inputs, y) in enumerate(self.train_loader):
                self.train_step(inputs, y)
                if i % print_freq == 0:
                    print(f'iter {i} loss : {self.loss_meter.avg}')
            print(f'@epoch {epoch} loss : {self.loss_meter.avg}')
            self.validate(epoch)
    
    def train_step(self, inputs, y):
        self.optim.zero_grad()

        inputs, y = inputs.to(self.device), y.to(self.device)
        logits = self.model(inputs)

        loss = self.criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optim.step()
        if self.scheduler:
            self.scheduler.step()
        self.loss_meter.update(loss.item())

        
    def validate(self, epoch):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        score_holder = []
        label_holder = []
        with torch.no_grad():
            for i, (inputs, y) in enumerate(self.val_loader):
                inputs, y = inputs.to(self.device), y.to(self.device)
                logits = self.model(inputs)
                loss = self.criterion(logits, y)
                loss_meter.update(loss.item())
                
                sim_score = self.criterion.get_target(logits)
                score_holder.append(sim_score.view(-1))
                label_holder.append(y.view(-1))

        sim_scores = torch.cat(score_holder, dim=0)
        labels = torch.cat(label_holder, dim=0)
        pr = pearson_r(sim_scores, labels)
        spearman = spearman_correlation(sim_scores, labels)

        print(f'val loss: {loss_meter.avg}, pearson_r: {pr.item()}, spearman: {spearman}')
        if self.use_wandb:
            wandb.log({
                "train_loss": self.loss_meter.avg,
                "val_loss": loss_meter.avg,
                "p corr": pr.item(),
                "spearman": spearman.item(),
            })

        if loss_meter.avg < self.val_best:
            self.val_best = loss_meter.avg 
            print('validation best..')
            self.save(self.val_best_path)

    def save(self, save_path):
        torch.save({
            'model_state': self.model.state_dict(),
            }, save_path)
        print(f'model saved at {save_path}')
    
    def load(self, load_path):
        save_dict = torch.load(load_path)
        self.model.load_state_dict(save_dict['model_state'])
        print(f'model loaded from {load_path}')
