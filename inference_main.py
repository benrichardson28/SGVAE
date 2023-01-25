import sys
import os
import os.path
import cmd_parser
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import inference_training as itr
import utils
import losses
import wandb
from datetime import datetime

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
    
    vae_config.data_path='~/cluster/fast/robot_grasp_data'
    tr,vl,ts = itr.latent_dataset_generator(inf_config,vae_config)
    

    lss_func = nn.MSELoss()
    inf_config.save_path = os.path.join(inf_config.vae_model_path,'inference')
    if not os.path.exists(inf_config.save_path):
        os.makedirs(inf_config.save_path)
    with open(f'{inf_config.save_path}/config.yaml', 'w') as conf_file:
        yaml.dump(inf_config, conf_file)

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


            inf_config.save_path = os.path.join(inf_config.save_path
                                                ,property,latent)
            






    return

if __name__ == '__main__':
    inf_config = cmd_parser.parse_inference_config(sys.argv[1:])
    with open(os.path.join(inf_config.vae_model_path,'config.yaml'), 'r') as f:
        vae_config = yaml.unsafe_load(f)
    main(inf_config,vae_config)


