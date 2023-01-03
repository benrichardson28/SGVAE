#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:19:51 2022

@author: richardson
"""

import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import networks as ntk
import utils

class Seq_MLVAE(pl.LightningModule):
    def __init__(self, model_hparams, opt_hparams, loss_hparams):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.encoder = ntk.MLVAE_Enc(**model_hparams)
        self.decoder = ntk.MLVAE_Enc(**model_hparams)
        self.content_sz = model_hparams.content_dim
        self.style_sz = model_hparams.style_dim
        
    def forward(self,):
        
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),**self.hparams.opt_hparams)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X,act,obj = batch
        context,style_mu,style_logvar = self.cns_init(X.shape[0],X.shape[1])
        
        for seq_ix in X.shape[1]:
            cm,clv,sm,slv,mu_x,lv_x = self.single_pass(X,seq_ix,context,style_mu,style_logvar)
            elbo,recon,kl_s,kl_c = self.loss_module(cm,clv,sm,slv,mu_x,lv_x,X)
            
            self.log("train loss", loss)
            
            style_mu[:,seq_ix] = sm.detach().clone()
            style_logvar[:,seq_ix] = slv.detach().clone()
        
        
        
    def validation_step(self, batch, batch_idx):
    def testing_step(self, batch, batch_idx):
        
    def single_pass(self,data,seq_ix,context,style_mu,style_logvar):
        sm,slv,cm,clv = self.encoder(X[:,seq_ix],context)
        #add on other styles, concat class
        
        class_mu = cm.unsqueeze(1).repeat([1,X.size(1),1])
        class_logvar = clv.unsqueeze(1).repeat([1,X.size(1),1])

        #reparam
        class_latent_embeddings = utils.reparameterize(training=training, mu=class_mu, logvar=class_logvar)   
        style_latent_embeddings = utils.reparameterize(training=training, mu=style_mu, logvar=style_logvar)
        
        #replace uninformed reparam with informed for this sequence
        style_latent_embeddings[:,seq_ix] = utils.reparameterize(training=training, mu=sm, logvar=slv)

        #reconstruct with action
        mu_x, logvar_x = self.decoder(style_latent_embeddings, class_latent_embeddings, action_batch)

        return cm,clv,sm,slv,mu_x,logvar_x
         
    # def add_model_specific_args(self,):
    def cns_init(self,batch_size,seq_len):
        cont_mu = torch.zeros((batch_size,self.content_sz),device=self.device)
        cont_lv = 0.5*torch.ones((batch_size,self.content_sz),device=self.device)
        
        style_mu = torch.zeros((batch_size,seq_len,self.style_sz),device=self.device)
        style_lv = 0.5*torch.ones((batch_size,seq_len,self.style_sz),device=self.device)
                            
        return torch.cat((cont_mu,cont_lv),dim=1),style_mu,style_lv
    
    def loss_module(self,cm,clv,sm,slv,mu_x,logvar_x,X):
        LOSS_PAR = self.hparams.loss_hparams
    
        #compute kl divergences for the different latent dimensions
        style_kl_divergence_loss = 0.5 * ( - 1 - slv + sm.pow(2) + slv.exp()).sum(dim=1)
        content_kl_divergence_loss = 0.5 * ( - 1 - clv + cm.pow(2) + clv.exp()).sum(dim=1)
        if LOSS_PAR.reduction='sum':
            style_kl_divergence_loss = style_kl_divergence_loss.sum()
            content_kl_divergence_loss = content_kl_divergence_loss.sum()
        elif LOSS_PAR.reduction=='mean':
            style_kl_divergence_loss = style_kl_divergence_loss.mean()
            content_kl_divergence_loss = content_kl_divergence_loss.mean()

        #### gaussian_beta_log_likelihood_loss(pred, target, beta=1):
        scale_x = (torch.exp(logvar_x) + 1e-12)#**0.5
        mean, var = torch.squeeze(mu_x,1),torch.squeeze(scale_x)
        logl = -0.5 * ((X - mean) ** 2 / var + torch.log(var) + math.log(2 * math.pi))
        weight = var.detach() ** LOSS_PAR.beta_nll
        logp_batch = torch.sum(logl * weight, axis=-1).sum(-1)
        if FLAGS.reduction=='sum':
            reconstruction_proba = logp_batch.sum()
        elif FLAGS.reduction=='mean':
            if style_weights is not None:
                logp_batch *= style_weights
            reconstruction_proba = logp_batch.sum(-1).mean()
        ########

        #combine all and apply betas
        total_kl = LOSS_PAR.style_coef*style_kl_divergence_loss + \
            LOSS_PAR.class_coef*class_kl_divergence_loss
        elbo = (reconstruction_proba - LOSS_PAR.beta_vae * total_kl)
        
        return elbo, reconstruction_proba, style_kl_divergence_loss, class_kl_divergence_loss
        
        
