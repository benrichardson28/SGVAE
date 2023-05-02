import copy
import torch
from torch.utils.data import DataLoader
import wandb

import pdb

import sgvae.training.sgvae_training as sgvae_training
import sgvae.utils as utils

def gen_latent(config,loader,encoder,action_names,dataset2build):
    with torch.no_grad():
        for _, (X, action_batch, obj_batch) in enumerate(loader):
            X = X.to(config.device)
            context,style_mu,style_logvar = utils.cNs_init(config,X.shape[0])
            dataset2build.start_row(obj_batch)
            for i in range(X.size(1)):
                sm,slv,cm,clv,=sgvae_training.single_pass(X, action_batch, i, context,
                                                        style_mu, style_logvar,
                                                        encoder, None, training=True)
                cont = torch.cat([cm,clv],dim=1).cpu()
                styl = torch.cat([sm,slv],dim=1).cpu()
                act_nms = action_names[torch.where(action_batch[:,-1-i]==1)[-1] % 4]
                dataset2build.add_to_row(cont,styl,act_nms,i)

                context = torch.cat([cm,clv],dim=1)

            dataset2build.append_row()
    return

def latent_dataset_generator(inf_config,vae_config):
    #### create VAE & initial dataset ####
    print('### Generate latent spaces for property inference ###')
    print('  Creating VAE models from checkpoint')
    encoder,decoder = utils.create_vae_models(vae_config)
    encoder,decoder,_,_,indices = utils.load_vae_checkpoint(inf_config.vae_model_path,
                                                      inf_config.vae_checkpoint,
                                                      encoder,decoder)
    encoder.to(vae_config.device)
    # decoder.to(vae_config.device)

    print('  Build datasets')
    vae_tr_set, vae_v_set, _, vae_ts_set = utils.create_vae_datasets(vae_config,indices,True)
    inf_tr_set,inf_v_set,inf_ts_set = utils.create_inference_datasets(vae_config)
    kwargs = {'num_workers': 1, 'pin_memory': True}

    for vaeds,infds in zip([vae_tr_set,vae_v_set,vae_ts_set],
                           [inf_tr_set,inf_v_set,inf_ts_set]):
        loader = DataLoader(vaeds,batch_size=vae_config.batch_size,**kwargs)
        for _ in range(inf_config.repetitions):
            vaeds.random_context_sampler()
            gen_latent(vae_config,loader,encoder,vaeds.act_list,infds)

    return inf_tr_set,inf_v_set,inf_ts_set

def process_epoch(config, model, loader, loss_func, optimizer=None):
    total_loss = 0
    for data,_,_,labels in loader:
  
        pred = model(data.to(config.device))
        batch_loss = loss_func(pred,labels.to(config.device))

        if optimizer:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_loss += batch_loss
    return total_loss.detach().cpu() / len(loader)


def train_model(config, model, loss_func, train_set, val_set,
                optimizer):
    train_loader = DataLoader(train_set,batch_size=config.batch_size,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=config.batch_size,shuffle=False)

    # for training & validation, only use final iteration
    train_set.iteration('last')
    val_set.iteration('last')

    # training
    # best_train_loss = 10000
    best_epoch = 0
    best_val_loss = 10000
    best_model = copy.deepcopy(model)

    for epoch in range(config.end_epoch):
        print('')
        print('Epoch #' + str(epoch) + '......................................................................')

        train_loss = process_epoch(config, model, train_loader, loss_func, optimizer)
        with torch.no_grad():
            val_loss = process_epoch(config, model, val_loader, loss_func)
        wandb.log({'Training Loss':train_loss,'Validation Loss':val_loss},step=epoch)

        if val_loss < best_val_loss:
            # best_train_loss = train_loss
            best_epoch = epoch
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

    utils.save_inf_checkpoint(config.save_path,best_epoch,best_model,optimizer)

    return best_model

def eval_model(config, model, loss_func, dataset):
    with torch.no_grad():
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=False)
        loss = process_epoch(config, model, loader, loss_func)
    return loss
