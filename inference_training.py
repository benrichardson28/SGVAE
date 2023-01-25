import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sgvae_training
import utils
import dataset_structs
import pdb

def gen_latent(FLAGS,loader,encoder,action_names,dataset2build):
    with torch.no_grad():
        for _, (X, action_batch, obj_batch) in enumerate(loader):
            X = X.to(FLAGS.device)
            context,style_mu,style_logvar = utils.cNs_init(FLAGS,X.shape[0])
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





