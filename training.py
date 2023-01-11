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
from ball_datasets import tactile_explorations,split_indices
import pdb
import matplotlib.pyplot as plt
import plotting

def loss(FLAGS,cm,clv,sm,slv,mu_x,logvar_x,X,style_weights=None):
    if FLAGS.reduction=='sum':
        style_kl_divergence_loss = 0.5 * ( - 1 - slv + sm.pow(2) + slv.exp()).sum()
        class_kl_divergence_loss = 0.5 * ( - 1 - clv + cm.pow(2) + clv.exp()).sum()
    elif FLAGS.reduction=='mean':
        style_kl_divergence_loss = 0.5 * ( - 1 - slv + sm.pow(2) + slv.exp()).sum(dim=1).mean()
        class_kl_divergence_loss = 0.5 * ( - 1 - clv + cm.pow(2) + clv.exp()).sum(dim=1).mean()

    #### gaussian_beta_log_likelihood_loss(pred, target, beta=1):
    scale_x = (torch.exp(logvar_x) + 1e-12)#**0.5
    mean, var = torch.squeeze(mu_x,1),torch.squeeze(scale_x)
    logl = -0.5 * ((X - mean) ** 2 / var + torch.log(var) + math.log(2 * math.pi))
    weight = var.detach() ** FLAGS.beta_nll

    if FLAGS.reduction=='sum':
        logp_batch = torch.sum(logl * weight, axis=-1).sum(-1)
        reconstruction_proba = logp_batch.sum()
    elif FLAGS.reduction=='mean':
        logp_batch = torch.sum(logl * weight, axis=-1).sum(-1)
        if style_weights is not None:
            logp_batch *= style_weights
        reconstruction_proba = logp_batch.sum(-1).mean()

    total_kl = FLAGS.style_coef*style_kl_divergence_loss + FLAGS.class_coef*class_kl_divergence_loss
    elbo = (reconstruction_proba - FLAGS.beta_VAE * total_kl)

    return elbo, reconstruction_proba, style_kl_divergence_loss, class_kl_divergence_loss

def process(FLAGS, X, action_batch, obj_batch, encoder, decoder, plot_params, act_list=None):
    context = torch.cat([torch.zeros(X.size(0),FLAGS.class_dim),
                         0.5*torch.ones(X.size(0),FLAGS.class_dim)],dim=1).to(FLAGS.device)

    style_mu = torch.zeros(X.size(0),X.size(1),FLAGS.style_dim).to(FLAGS.device)
    style_logvar = 0.5*torch.ones(X.size(0),X.size(1),FLAGS.style_dim).to(FLAGS.device)

    X = X.to(FLAGS.device)
    action_batch=action_batch.to(FLAGS.device)

    losses = {'elbo':[],
              'recon_prob':[],
              'kl_class':[],
              'kl_style':[]}
    total_elbo = 0
    ixs = np.random.choice(range(X.size(0)),4,replace=False)
    # context loop
    for cs in range(X.size(1)):
        #pass in first sample -> get content and style
        sm,slv,cm,clv,mu_x,logvar_x = single_pass(X,action_batch,cs,context,style_mu,style_logvar,encoder,decoder)

        style_weights = None
        if FLAGS.weight_style:
            style_weights = torch.ones(X.size(0),X.size(1))
            style_weights = style_weights.to(FLAGS.device)
            style_weights[:,:-1-cs] *= 0.5

        elbo, reconstruction_proba, style_kl_divergence_loss, \
            class_kl_divergence_loss = loss(FLAGS,cm,clv,sm,slv,mu_x,logvar_x,X,style_weights)

        losses['elbo'].append(elbo)
        losses['recon_prob'].append(reconstruction_proba)
        losses['kl_class'].append(class_kl_divergence_loss)
        losses['kl_style'].append(style_kl_divergence_loss)
        total_elbo += elbo*1.

        # prepare latents for next pass: create context
        context = torch.cat([cm,clv],dim=1)

        # if plot_params['plot']:
        #     x1 = X[ixs].cpu()
        #     x2 = mu_x[ixs].detach().cpu()
        #     acts = action_batch[ixs].cpu()
        #     objs = obj_batch[ixs].cpu()
        #     plotting.plot_multiple(x1,x2,acts,objs,cs,plot_params,act_list,FLAGS.logdir)

    losses['elbo'] = torch.stack(losses['elbo'])
    losses['recon_prob'] = torch.stack(losses['recon_prob'])
    losses['kl_class'] = torch.stack(losses['kl_class'])
    losses['kl_style'] = torch.stack(losses['kl_style'])

    return total_elbo, losses

def single_pass(X,action_batch,cs,context,style_mu,style_logvar,encoder,decoder,training=True):
    #style vars should be updated outside the function

    sm,slv,cm,clv = encoder(X[:,-1-cs],context)
    #add on other styles, concat class
    style_mu[:,-1-cs] = sm.detach().clone()
    style_logvar[:,-1-cs] = slv.detach().clone()
    class_mu = cm.unsqueeze(1).repeat([1,X.size(1),1])
    class_logvar = clv.unsqueeze(1).repeat([1,X.size(1),1])

    #reparam
    class_latent_embeddings = reparameterize(training=training, mu=class_mu, logvar=class_logvar)   #batch x 4 x 10
    single_style_latent = reparameterize(training=training, mu=sm, logvar=slv)   #batch x 10
    style_latent_embeddings = reparameterize(training=training, mu=style_mu, logvar=style_logvar)

    style_latent_embeddings[:,-1-cs] = single_style_latent

    #reconstruct with action
    mu_x, logvar_x = decoder(style_latent_embeddings, class_latent_embeddings, action_batch)

    return sm,slv,cm,clv,mu_x,logvar_x


def eval(FLAGS, valid_loader, encoder, decoder):
    plot_params = {'plot':False}
    elbo_epoch = 0
    term1_epoch = torch.zeros(4)
    term2_epoch = torch.zeros(4)
    term3_epoch = torch.zeros(4)
    term4_epoch = torch.zeros(4)
    losses = {'elbo':[],
              'recon_prob':[],
              'kl_class':[],
              'kl_style':[]}
    with torch.no_grad():
        for it, (data_batch, action_batch, obj_batch) in enumerate(valid_loader):

            elbo, component_loss = process(FLAGS, data_batch, action_batch, obj_batch, encoder, decoder, plot_params)

            elbo_epoch += elbo*1.
            term1_epoch += component_loss['elbo'].detach().cpu()
            term2_epoch += component_loss['recon_prob'].detach().cpu()
            term3_epoch += component_loss['kl_class'].detach().cpu()
            term4_epoch += component_loss['kl_style'].detach().cpu()

        elbo_epoch /= (it + 1)
        term1_epoch /= (it + 1)
        term2_epoch /= (it + 1)
        term3_epoch /= (it + 1)
        term4_epoch /= (it + 1)

    return elbo_epoch,term1_epoch,term2_epoch,term3_epoch,term4_epoch

def training_procedure(FLAGS):
    FLAGS.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    model definition
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim,
                      in_channels = FLAGS.in_channels,
                      hidden_dims = FLAGS.hidden_dims,
                      kernels = FLAGS.kernels,
                      strides = FLAGS.strides,
                      paddings = FLAGS.paddings).to(FLAGS.device)
    #encoder.apply(weights_init)
    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim,
                      hidden_dims = FLAGS.hidden_dims,
                      out_channels = FLAGS.in_channels,
                      kernels = FLAGS.kernels,
                      strides = FLAGS.strides,
                      paddings = FLAGS.paddings,
                      cos = encoder.cos).to(FLAGS.device)

    """
    optimizer definition
    """
    auto_encoder_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    print('Loading ball EP dataset...')
    ds_train = tactile_explorations(train=True,dataset=FLAGS.dataset)
    ds_val = tactile_explorations(train=True,dataset=FLAGS.dataset)
    # Creating data indices for training and validation splits:
    train_indices, val_indices = split_indices(ds_train,6,FLAGS.dataset)

    #set indices in datasets, compute and apply normalizations
    ds_train.set_indices(train_indices)
    ds_train.set_transform()
    ds_val.set_indices(val_indices)
    ds_val.set_transform(ds_train.get_transform())

    #dataloader prep
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader = DataLoader(ds_train,batch_size=FLAGS.batch_size,shuffle=True,**kwargs)
    val_loader = DataLoader(ds_val,batch_size=FLAGS.batch_size,**kwargs)

    # losses
    layout = {
        "Loss": {
            #"summed elbo": ["summed elbo"],
            "context elbo": ["Multiline", ["ce1", "ce2", "ce3", "ce4"]],
            "context recon_prob": ["Multiline", ["cr1", "cr2", "cr3", "cr4"]],
            "context KL class": ["Multiline", ["cc1", "cc2", "cc3", "cc4"]],
            "context KL style": ["Multiline", ["cs1", "cs2", "cs3", "cs4"]],
        },
        "Validation loss": {
            #"summed elbo": ["summed elbo"],
            "context elbo": ["Multiline", ["ve1", "ve2", "ve3", "ve4"]],
            "context recon_prob": ["Multiline", ["vr1", "vr2", "vr3", "vr4"]],
            "context KL class": ["Multiline", ["vc1", "vc2", "vc3", "vc4"]],
            "context KL style": ["Multiline", ["vs1", "vs2", "vs3", "vs4"]],
        },
    }
    writer = SummaryWriter(comment=FLAGS.save_path)
    writer.add_custom_scalars(layout)
    FLAGS.logdir=writer.logdir


    with open(f'{FLAGS.logdir}/../run_tracker.csv', mode='a+',newline='') as tracking_file:
        meta_track = csv.DictWriter(tracking_file,fieldnames=['Output_folder',*vars(FLAGS).keys()])
        if len(open(f'{FLAGS.logdir}/../run_tracker.csv', 'r').readlines())<=1:
            meta_track.writeheader()
        hlink = f'=HYPERLINK("/home/richardson/cluster/robot_haptic_perception/{FLAGS.logdir}")'
        meta_track.writerow({'Output_folder':hlink,**vars(FLAGS)})
    with open(f'{writer.logdir}/config.yaml', 'w') as conf_file:
        yaml.dump(FLAGS, conf_file)
    savedir = f'{writer.logdir}/checkpoints_{FLAGS.batch_size}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    monitor = torch.zeros(FLAGS.end_epoch - FLAGS.start_epoch)

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('')
        print('Epoch #' + str(epoch) + '......................................................................')
        elbo_epoch = 0
        term1_epoch = torch.zeros(4)
        term2_epoch = torch.zeros(4)
        term3_epoch = torch.zeros(4)
        term4_epoch = torch.zeros(4)

        #refresh context here? don't need to change anything about the dataloader
        # generate the random contexts in the dataset
        ds_train.random_context_sampler()
        ds_val.random_context_sampler()

        for it, (data_batch, action_batch, obj_batch) in enumerate(loader):
            # set zero_grad for the optimizer
            auto_encoder_optimizer.zero_grad()
            X = data_batch.to(FLAGS.device).detach().clone()

            ps=False
            if (((epoch+1) in [1,50,250,1000]) and ((it+1)%len(loader)==0)):
                 ps=True
            plot_param = {'plot':ps, 'epoch':epoch,
                          'cols':ds_train.col_order}
            elbo, component_loss = process(FLAGS, X, action_batch, obj_batch,
                                           encoder, decoder, plot_param, ds_train.act_list)
            (-elbo).backward()
            auto_encoder_optimizer.step()
            elbo_epoch += elbo.detach().cpu()
            writer.add_scalar('batch_loss elbo',component_loss['elbo'][-1].detach().cpu(),it)
            term1_epoch += component_loss['elbo'].detach().cpu()
            term2_epoch += component_loss['recon_prob'].detach().cpu()
            term3_epoch += component_loss['kl_class'].detach().cpu()
            term4_epoch += component_loss['kl_style'].detach().cpu()
            print(f'\r{it}/{len(loader)}',end='')
        print("\nElbo epoch %.2f" % (elbo_epoch.item() / (it + 1)))
        print("Rec. Proba %.2f" % (term2_epoch[-1] / (it + 1)))
        print("KL style %.2f" % (term4_epoch[-1] / (it + 1)))
        print("KL content %.2f" % (term3_epoch[-1] / (it + 1)))

        writer.add_scalar('Loss/summed elbo', elbo_epoch/ (it + 1) ,epoch)
        for i in range(len(term1_epoch)):
            writer.add_scalar(f'ce{i+1}',term1_epoch[i]/(it + 1),epoch)
            writer.add_scalar(f'cr{i+1}',term2_epoch[i]/(it + 1),epoch)
            writer.add_scalar(f'cc{i+1}',term3_epoch[i]/(it + 1),epoch)
            writer.add_scalar(f'cs{i+1}',term4_epoch[i]/(it + 1),epoch)

        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            elbo,te1,te2,te3,te4=eval(FLAGS, val_loader, encoder, decoder)
            monitor[epoch]=elbo
            writer.add_scalar('Validation loss/summed elbo', elbo ,epoch)
            for i in range(len(term1_epoch)):
                writer.add_scalar(f've{i+1}',te1[i],epoch)
                writer.add_scalar(f'vr{i+1}',te2[i],epoch)
                writer.add_scalar(f'vc{i+1}',te3[i]/(it + 1),epoch)
                writer.add_scalar(f'vs{i+1}',te4[i]/(it + 1),epoch)

            torch.save(monitor, os.path.join(savedir, 'monitor_e%d'%epoch))
            torch.save(encoder.state_dict(), os.path.join(savedir, FLAGS.encoder_save +'_e%d'%epoch))
            torch.save(decoder.state_dict(), os.path.join(savedir, FLAGS.decoder_save +'_e%d'%epoch))
