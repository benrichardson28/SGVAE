import sys
import os
import os.path
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from sgvae.networks import Property_model
import sgvae.cmd_parser as cmd_parser
import sgvae.inference.inference_training as itr

import pdb


def main(inf_config,vae_config):
    if torch.cuda.is_available() and (inf_config.device=='gpu'):
        print('Cuda available; running on gpu.')
        inf_config.device = torch.device('cuda')
        vae_config.device = torch.device('cuda')
    else:
        print('Cuda unavailable; running on cpu.')
        inf_config.device = torch.device('cpu')
        vae_config.device = torch.device('cpu')

    #vae_config.data_path='~/cluster/fast/robot_grasp_data'
    base_path = os.path.join(inf_config.vae_model_path,'inference',inf_config.save_path)
    inf_config.save_path = base_path[:]
    
    os.makedirs(inf_config.save_path,exist_ok=True)
    os.makedirs(os.path.join(inf_config.save_path,'models'),exist_ok=True)

    tr,vl,ts = itr.latent_dataset_generator(inf_config,vae_config)

    results_df = pd.DataFrame(columns = ['Property','Latent Type',
                                        'Iteration','Dataset Type',
                                        'MLE Loss'])

    for property in inf_config.test_properties:
        tr.label(property)
        vl.label(property)
        ts.label(property)

        if 'contents' in property:
            loss_func = nn.CrossEntropyLoss()
            class_cnt = tr.class_cnt
        else:
            loss_func = nn.MSELoss()
            class_cnt = 1

        latent_options = ['content']
        if vae_config.style_dim > 0: latent_options.extend(['style'])
        for latent in latent_options:
            # writer and tracker
            wandb_id = wandb.util.generate_id()
            wandb.init(config=vae_config,project="SGVAE_inference", entity="brichardson",id=wandb_id,
                       )
            wandb.config.update(inf_config,allow_val_change=True)
            wandb.config.update({"property":property,"latent":latent})

            tr.latent_type(latent)
            vl.latent_type(latent)
            ts.latent_type(latent)

            inf_config.save_path = os.path.join(base_path,'models',
                                                f'{property}_{latent}')

            if latent=='content': dim = 2*vae_config.content_dim
            if latent=='style': dim = 2*vae_config.style_dim
            model = Property_model(z_dim=dim, num_classes=class_cnt).to(inf_config.device)
            optimizer = optim.Adam(
                list(model.parameters()),
                lr=inf_config.initial_learning_rate,
                betas=(inf_config.beta_1, inf_config.beta_2),
                weight_decay=inf_config.weight_decay
            )

            best_model = itr.train_model(inf_config, model,
                                         loss_func, tr, vl,
                                         optimizer)

            for iter in range(tr.sequence_len):
                for d_set,d_name in zip([tr,vl,ts],['Train','Val','Test']):
                    d_set.iteration(iter)
                    loss = itr.eval_model(inf_config,best_model,loss_func,d_set)
                    res_row = {'Property':property,'Latent Type':latent,
                               'Iteration':iter,'Dataset Type':d_name,
                               'MLE Loss':loss}
                    results_df = results_df.append(res_row,ignore_index=True)

            # wrap up
            wandb.finish()
            res_path = os.path.join(base_path,'results.csv')
            results_df.to_csv(res_path,mode='a',header=not os.path.exists(res_path))
    return

if __name__ == '__main__':
    inf_config = cmd_parser.parse_inference_config(sys.argv[1:])
    with open(os.path.join(inf_config.vae_model_path,'config.yaml'), 'r') as f:
        vae_config = yaml.unsafe_load(f)
    main(inf_config,vae_config)
