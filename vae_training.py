import os
import os.path
import random
import numpy as np
import math
import yaml
import csv

import torch
import torch.optim as optim
from torch.distributions import Normal, Bernoulli, MultivariateNormal, kl
from utils import weights_init
from networks import Encoder, Decoder
from torch.utils.data import DataLoader
from utils import reparameterize
from torch.nn import functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataset_structs import tactile_explorations,split_indices
import pdb
import matplotlib.pyplot as plt
import plotting
import utils

def loss(FLAGS,cm,clv,sm,slv,mu_x,logvar_x,X,style_weights=None):
    if FLAGS.reduction=='sum':
        style_kl_divergence_loss = 0.5 * ( - 1 - slv + sm.pow(2) + slv.exp()).sum()
        content_kl_divergence_loss = 0.5 * ( - 1 - clv + cm.pow(2) + clv.exp()).sum()
    elif FLAGS.reduction=='mean':
        style_kl_divergence_loss = 0.5 * ( - 1 - slv + sm.pow(2) + slv.exp()).sum(dim=1).mean()
        content_kl_divergence_loss = 0.5 * ( - 1 - clv + cm.pow(2) + clv.exp()).sum(dim=1).mean()

    #### gaussian_beta_log_likelihood_loss(pred, target, beta=1):
    scale_x = (torch.exp(logvar_x) + 1e-12)#**0.5
    mean, var = torch.squeeze(mu_x,1),torch.squeeze(scale_x)
    logl = -0.5 * ((X - mean) ** 2 / var + torch.log(var) + math.log(2 * math.pi))
    weight = var.detach() ** FLAGS.beta_NLL

    if FLAGS.reduction=='sum':
        logp_batch = torch.sum(logl * weight, axis=-1).sum(-1)
        reconstruction_proba = logp_batch.sum()
    elif FLAGS.reduction=='mean':
        logp_batch = torch.sum(logl * weight, axis=-1).sum(-1)
        if style_weights is not None:
            logp_batch *= style_weights
        reconstruction_proba = logp_batch.mean(-1).mean()
    #reconstruction_proba /= (FLAGS.action_repetitions * 4)

    total_kl = FLAGS.style_coef*style_kl_divergence_loss + FLAGS.content_coef*content_kl_divergence_loss
    elbo = (reconstruction_proba - FLAGS.beta_VAE * total_kl)

    return elbo, reconstruction_proba, style_kl_divergence_loss, content_kl_divergence_loss

def process(FLAGS, X, action_batch, encoder, decoder, loss_logger):
    context, style_mu, style_logvar = utils.cNs_init(FLAGS)
    X = X.to(FLAGS.device)
    action_batch=action_batch.to(FLAGS.device)

    total_elbo = 0
    # context loop
    for cs in range(X.size(1)):
        #pass in first sample -> get content and style
        sm,slv,cm,clv,mu_x,logvar_x = single_pass(X,action_batch,cs,context,style_mu,style_logvar,encoder,decoder)

        style_weights = None
        if FLAGS.weight_style:
            style_weights = torch.ones(X.size(0),X.size(1))
            style_weights = style_weights.to(FLAGS.device)
            style_weights[:,:-1-cs] *= 0.5

        elbo, mle, kl_style, \
            kl_content = loss(FLAGS,cm,clv,sm,slv,mu_x,logvar_x,X,style_weights)
        total_elbo += elbo*1.
        loss_logger.update_epoch_loss(elbo,mle,kl_content,kl_style,cs)

        # prepare latents for next pass: create context
        context = torch.cat([cm,clv],dim=1)

    return total_elbo / (FLAGS.action_repetitions*4)

def single_pass(X,action_batch,cs,context,style_mu,style_logvar,encoder,decoder,training=True):
    #style vars should be updated outside the function

    sm,slv,cm,clv = encoder(X[:,-1-cs],context)
    #add on other styles, concat content
    style_mu[:,-1-cs] = sm.detach().clone()
    style_logvar[:,-1-cs] = slv.detach().clone()
    content_mu = cm.unsqueeze(1).repeat([1,X.size(1),1])
    content_logvar = clv.unsqueeze(1).repeat([1,X.size(1),1])

    #reparam
    content_latent_embeddings = reparameterize(training=training, mu=content_mu, logvar=content_logvar)   #batch x 4 x 10
    single_style_latent = reparameterize(training=training, mu=sm, logvar=slv)   #batch x 10
    style_latent_embeddings = reparameterize(training=training, mu=style_mu, logvar=style_logvar)

    style_latent_embeddings[:,-1-cs] = single_style_latent

    #reconstruct with action
    mu_x, logvar_x = decoder(style_latent_embeddings, content_latent_embeddings, action_batch)

    return sm,slv,cm,clv,mu_x,logvar_x


def eval(FLAGS, encoder, decoder, loader, logger, epoch):

    with torch.no_grad():
        for it, (data_batch, action_batch, obj_batch) in enumerate(loader):

            elbo = process(FLAGS, data_batch, action_batch, 
                           encoder, decoder, logger)

    logger.finalize_epoch_loss(it+1)
    logger.logwandb(epoch)

def train_epoch(FLAGS,encoder,decoder,optimizer,
                train_loader,train_logger,
                epoch):

    for it, (data_batch, action_batch, obj_batch) in enumerate(train_loader):
        # set zero_grad for the optimizer
        optimizer.zero_grad()
        X = data_batch.to(FLAGS.device).detach().clone()

        elbo = process(FLAGS, X, action_batch,
                        encoder, decoder, train_logger)
        (-elbo).backward()
        optimizer.step()

        print(f'\r{it}/{len(train_loader)}',end='')

    train_logger.finalize_epoch_loss(it+1)
    train_logger.print_losses()
    train_logger.logwandb(epoch)