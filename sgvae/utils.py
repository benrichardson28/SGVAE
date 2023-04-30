import os
import re
from argparse import Namespace
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms as transforms
import pdb
import numpy as np
from torchvision import datasets
from networks import Encoder,Decoder
import dataset_structs as dst
import json

def create_vae_models(FLAGS):
    """
    model definitions
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, 
                      content_dim=FLAGS.content_dim,
                      in_channels = FLAGS.in_channels,
                      hidden_dims = FLAGS.hidden_dims,
                      kernels = FLAGS.kernels,
                      strides = FLAGS.strides,
                      paddings = FLAGS.paddings,
                    )#remove_context = FLAGS.update_prior)
    #encoder.apply(weights_init)
    decoder = Decoder(style_dim=FLAGS.style_dim, 
                      content_dim=FLAGS.content_dim,
                      hidden_dims = FLAGS.hidden_dims,
                      out_channels = FLAGS.in_channels,
                      kernels = FLAGS.kernels,
                      strides = FLAGS.strides,
                      paddings = FLAGS.paddings,
                      cos = encoder.cos)
    #decoder.apply(weights_init)
    return encoder,decoder 

def create_vae_optimizer(FLAGS,encoder,decoder):
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2),
        eps=10e-4
    )
    return optimizer

def create_vae_scheduler():
    return NotImplementedError

def create_vae_datasets(FLAGS,indices=None,test=False):
    train_set = dst.tactile_explorations(FLAGS,train=True,
                                         dataset=FLAGS.dataset)
    validation_set = copy.deepcopy(train_set)
    if indices is None:
        train_indices, val_indices = dst.split_indices(train_set,
                                                       FLAGS.split_ratio,
                                                       FLAGS.dataset)
    else:
        train_indices, val_indices = indices
    train_set.set_indices(train_indices)
    train_set.set_transform()
    validation_set.set_indices(val_indices)
    validation_set.set_transform(train_set.get_transform())

    test_set = None
    if test:
        test_set = dst.tactile_explorations(FLAGS,train=False,
                                            dataset=FLAGS.dataset)
        test_set.set_transform(train_set.get_transform())

    return train_set, validation_set, [train_indices,val_indices], test_set

def create_inference_datasets(iters):
    train_set = dst.latent_representations(iters)
    validation_set = dst.latent_representations(iters)
    test_set = dst.latent_representations(iters)
    return train_set,validation_set,test_set

def cNs_init(FLAGS,size):
    act_num = 4 * FLAGS.action_repetitions
    context = torch.zeros(size,2*FLAGS.content_dim).to(FLAGS.device)
    style_mu = torch.zeros(size,act_num,FLAGS.style_dim).to(FLAGS.device)
    style_logvar = torch.zeros(size,act_num,FLAGS.style_dim).to(FLAGS.device)
    return context, style_mu, style_logvar

def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = torch.zeros_like(std).normal_()
        return eps.mul(std).add_(mu)
    else:
        return mu

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
                        scheduler=None):
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

def weights_init(layer):
    r"""Apparently in Chainer Lecun normal initialisation was the default one
    """
    if isinstance(layer, nn.Linear):
        lecun_normal_(layer.bias)
        lecun_normal_(layer.weight)

def lecun_normal_(tensor, gain=1):

    import math
    r"""Adapted from https://pytorch.org/docs/0.4.1/_modules/torch/nn/init.html#xavier_normal_
    """
    dimensions = tensor.size()
    if len(dimensions) == 1:  # bias
        fan_in = tensor.size(0)
    elif len(dimensions) == 2:  # Linear
        fan_in = tensor.size(1)
    else:
        num_input_fmaps = tensor.size(1)
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size

    std = gain * math.sqrt(1.0 / (fan_in))
    with torch.no_grad():
        return tensor.normal_(0, std)

def imshow_grid(images, shape=[2, 8], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    if save:
        plt.savefig('reconstructed_images/' + str(name) + '.png')
        plt.clf()
    else:
        plt.show()