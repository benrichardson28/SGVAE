import os
import re
import json
import copy
import torch
import torch.optim as optim

from sgvae.networks import Encoder,Decoder
import sgvae.dataset_structs as dst
from sgvae import EXPLORATORY_PROCEDURE_NUM

import pdb



# Model and training procedure
def create_vae_models(config):
    """ Create encoder and decoder modules for VAE.

    :param Namespace config:
        * style_dim (int) -- Size of style latent space.
        * content_dim (int) -- Size of content latent space.
        * in_channels (int) -- Number of channels in the data.
        * hidden_dims (list) -- Number of kernels to use at each layer.
        * kernels (list) -- Convolution kernel sizes for each layer.
        * strides (list) -- Stride lengths for kernels at each layer.
        * paddings (list) -- Padding to apply to input for each layer.
    """
    encoder = Encoder(style_dim=config.style_dim, 
                      content_dim=config.content_dim,
                      in_channels = config.in_channels,
                      hidden_dims = config.hidden_dims,
                      kernels = config.kernels,
                      strides = config.strides,
                      paddings = config.paddings,
                    )#remove_context = config.update_prior)
    #encoder.apply(weights_init)
    decoder = Decoder(style_dim=config.style_dim, 
                      content_dim=config.content_dim,
                      data_len = encoder.data_len,
                      hidden_dims = config.hidden_dims,
                      out_channels = config.in_channels,
                      kernels = config.kernels,
                      strides = config.strides,
                      paddings = config.paddings)
    #decoder.apply(weights_init)
    return encoder,decoder 

def create_vae_optimizer(config,encoder,decoder):
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.initial_learning_rate,
        betas=(config.beta_1, config.beta_2),
        eps=10e-4
    )
    return optimizer

def create_vae_scheduler():
    return NotImplementedError



# Dataset creators
def create_vae_datasets(config,indices=None,test=False):
    train_set = dst.tactile_explorations(config,train=True,
                                         dataset=config.dataset)
    validation_set = copy.deepcopy(train_set)
    if indices is None:
        train_indices, val_indices = dst.split_indices(train_set,
                                                       config.split_ratio,
                                                       config.dataset)
    else:
        train_indices, val_indices = indices
    train_set.set_indices(train_indices)
    train_set.set_transform()
    validation_set.set_indices(val_indices)
    validation_set.set_transform(train_set.get_transform())

    test_set = None
    if test:
        test_set = dst.tactile_explorations(config,train=False,
                                            dataset=config.dataset)
        test_set.set_transform(train_set.get_transform())

    return train_set, validation_set, [train_indices,val_indices], test_set

def create_inference_datasets(action_repetitions,style_dim,prop_path):
    trn = dst.latent_representations(action_repetitions,style_dim,prop_path)
    val = dst.latent_representations(action_repetitions,style_dim,prop_path)
    tst = dst.latent_representations(action_repetitions,style_dim,prop_path)
    return trn,val,tst



# Vae training and inference functions
def cNs_init(config,size):
    act_num = EXPLORATORY_PROCEDURE_NUM * config.action_repetitions
    context = torch.zeros(size,2*config.content_dim).to(config.device)
    style_mu = torch.zeros(size,act_num,config.style_dim).to(config.device)
    style_logvar = torch.zeros(size,act_num,config.style_dim).to(config.device)
    return context, style_mu, style_logvar

def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = torch.zeros_like(std).normal_()
        return eps.mul(std).add_(mu)
    else:
        return mu


# Checkpointing functions
def save_vae_checkpoint(folder,epoch,wandb_id,
                        encoder,decoder,
                        optimizer=None,scheduler=None
                        ):
    checkpoint = { 
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_sched': scheduler.state_dict() if scheduler is not None else None,
        'wandb_id': wandb_id
        #'loss_logger': loss_logger}
    }
    torch.save(checkpoint, os.path.join(folder,f'checkpoint_{epoch}.pth'))

def checkpoint_exists(folder):
    checkpoint_folder = os.path.join(folder,'checkpoints')
    if not os.path.isdir(checkpoint_folder): return False
    if len(os.listdir(checkpoint_folder)) == 0: return False
    max_epoch=0
    for checkpoint in os.listdir(checkpoint_folder):
        ep = re.findall(r"\d+",checkpoint)
        max_epoch = max(int(ep[0]),max_epoch)
    return max_epoch
    
def load_vae_checkpoint(device,
                        folder,
                        epoch,
                        encoder,
                        decoder,
                        optimizer=None,
                        scheduler=None,
                        wandb_id=None):
    print("Loading checkpoint")
    filename = os.path.join(folder,'checkpoints',f'checkpoint_{epoch}.pth')
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        wandb_id = checkpoint['wandb_id']
        #loss_logger = checkpoint(['loss_logger'])
    with open(os.path.join(folder,'split_indices.json')) as f:
        indices = json.load(f)
    return encoder, decoder, optimizer, scheduler, indices, wandb_id

def save_inf_checkpoint(file_name,epoch,model,
                        optimizer=None,scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_sched': scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(checkpoint, file_name + f'_{epoch}.pth')

