import pandas as pd
import torch
from tensorboardX import SummaryWriter
import wandb


class vae_loss_logger():
    def __init__(self,FLAGS,stage):
        self.iterations = FLAGS.action_repetitions * 4
        self.reset_epoch_loss()
        self.stage = stage
        #self.table = wandb.Table(columns=["Total","ELBO","NLL","KL Content","KL Style"])

    def reset_epoch_loss(self):
        self.total_loss = 0
        self.elbo = torch.zeros(self.iterations)
        self.mle = torch.zeros(self.iterations)
        self.kl_content = torch.zeros(self.iterations)
        self.kl_style = torch.zeros(self.iterations)

    def update_epoch_loss(self,elbo,mle,klc,kls,iter_ix):
        self.total_loss += elbo.detach().cpu()
        self.elbo[iter_ix] += elbo.detach().cpu()
        self.mle[iter_ix] += mle.detach().cpu()
        self.kl_content[iter_ix] += klc.detach().cpu()
        self.kl_style[iter_ix] += kls.detach().cpu()

    def finalize_epoch_loss(self,count):
        self.total_loss /= self.iterations
        self.total_loss /= count
        self.elbo /= count
        self.mle /= count
        self.kl_content /= count
        self.kl_style /= count

    def print_losses(self):
        print("\n")
        print(f'{self.stage} Losses:')
        print(f"    Total loss: {self.total_loss:.2f}")
        print(f"    ELBO loss: {self.elbo[-1]:.2f}")
        print(f"    MLE loss: {self.mle[-1]:.2f}")
        print(f"    KL Content loss: {self.kl_content[-1]:.2f}")
        print(f"    KL Style loss: {self.kl_style[-1]:.2f}")

    def logwandb(self,epoch):
        fl=self.stage
        
        #wandb.log({fl'})
        loss_dict = {f'{fl} Total': self.total_loss,
                     f'{fl}/ELBO': {i:self.elbo[i] for i in range(self.iterations)},
                     f'{fl}/MLE': {i:self.mle[i] for i in range(self.iterations)},
                     f'{fl}/KL Content': {i:self.kl_content[i] for i in range(self.iterations)},
                     f'{fl}/KL Style': {i:self.kl_style[i] for i in range(self.iterations)}}
        wandb.log(loss_dict,step=epoch)