import sys
import os
import os.path
import cmd_parser
import yaml
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import inference_training as itr
import utils
import logger
import wandb
from datetime import datetime
import networks

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
    
    pdb.set_trace()
    vae_config.data_path='~/cluster/fast/robot_grasp_data'
    tr,vl,ts = itr.latent_dataset_generator(inf_config,vae_config)
    pdb.set_trace()
    inf_config.save_path = os.path.join(inf_config.vae_model_path,'inference')
    base_path = os.path.join(inf_config.vae_model_path,'inference')
    if not os.path.exists(inf_config.save_path):
        os.makedirs(inf_config.save_path)
    # with open(f'{inf_config.save_path}/config.yaml', 'w') as conf_file:
    #     yaml.dump(inf_config, conf_file)

    results_df = pd.DataFrame(columns = ['Property','Latent Type',
                                        'Iteration','Dataset Type',
                                        'MLE Loss'])

    for property in inf_config.test_properties:
        tr.set_lbl(property)
        vl.set_lbl(property)
        ts.set_lbl(property)

        if 'contents' in property:
            loss_func = nn.CrossEntropyLoss()
            class_cnt = tr.get_class_cnt()
        else:
            loss_func = nn.MSELoss()
            class_cnt = 1

        latent_options = ['content']
        if vae_config.style_dim > 0: latent_options.extend(['style'])
        for latent in latent_options:
            # writer and tracker
            wandb_id = wandb.util.generate_id()
            wandb.init(config=inf_config,project="SGVAE", entity="brichardson",id=wandb_id,
                       group='inference',tags=[property,latent])

            #config.save_path=writer.logdir
            #inf_config.save_path = os.path.join('runs',config.save_path)
            #train_loss_logger = losses.vae_loss_logger(config,'Training')
            #val_loss_logger = losses.vae_loss_logger(config,'Validation')

            tr.set_latent(latent)
            vl.set_latent(latent)
            ts.set_latent(latent)

            inf_config.save_path = os.path.join(base_path,'models',
                                                f'{property}_{latent}_')
            
            if latent=='content': dim = 2*vae_config.content_dim
            if latent=='style': dim = 2*vae_config.style_dim
            model = networks.Property_model(z_dim=dim, num_classes=class_cnt).to(inf_config.device)
            optimizer = optim.Adam(
                list(model.parameters()),
                lr=inf_config.initial_learning_rate,
                betas=(inf_config.beta_1, inf_config.beta_2),
                weight_decay=inf_config.weight_decay
            )           

            best_model = itr.train_model(inf_config, model, 
                                         loss_func, tr, vl, 
                                         optimizer) 
            
            for iter in tr.sequence_len:
                for d_set,d_name in zip([tr,vl,ts],['Train','Val','Test']):
                    d_set.set_iteration(iter)
                    loss = itr.eval_model(inf_config,best_model,loss_func,d_set)
                    res_row = {'Property':property,'Latent Type':latent,
                               'Iteration':iter,'Dataset Type':d_name,
                               'MLE Loss':loss}
                    results_df = results_df.append(res_row,ignore_index=True)
            
            # wrap up
            wandb.finish()
    results_df.to_csv(os.path.join(inf_config.save_path,'results.csv'))
    return

if __name__ == '__main__':
    inf_config = cmd_parser.parse_inference_config(sys.argv[1:])
    with open(os.path.join(inf_config.vae_model_path,'config.yaml'), 'r') as f:
        vae_config = yaml.unsafe_load(f)
    main(inf_config,vae_config)


