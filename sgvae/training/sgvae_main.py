import sys
import os
import os.path
import time
import wandb
import json
import torch
from torch.utils.data import DataLoader

import sgvae.cmd_parser as cmd_parser
import sgvae.training.sgvae_training as trl
import sgvae.utils as utils
import sgvae.logger as logger

import pdb


def main(config):
    start_time = time.perf_counter()
    print(start_time)

    if torch.cuda.is_available() and (config.device=='gpu'):
        print('Cuda available; running on gpu.')
        config.device = torch.device('cuda')
    else: 
        print('Cuda unavailable; running on cpu.')
        config.device = torch.device('cpu')

    # models and optimizer
    print('Creating models.')
    encoder,decoder = utils.create_vae_models(config)
    encoder.to(config.device)
    decoder.to(config.device)
    optimizer = utils.create_vae_optimizer(config,encoder,decoder)

    indices, wandb_id = None, None
    checkpoint = utils.checkpoint_exists(config.save_path)
    if checkpoint:
        encoder,decoder,\
            optimizer,_,indices,\
                 wandb_id = utils.load_vae_checkpoint(config.device,
                                                      config.save_path,
                                                      checkpoint,
                                                      encoder,
                                                      decoder,
                                                      optimizer)

    # datasets
    print('Creating datasets.')
    training_set, validation_set, indices, _ = utils.create_vae_datasets(config,indices)

    # dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(training_set,batch_size=config.batch_size,
                        shuffle=True,**kwargs)
    val_loader = DataLoader(validation_set,batch_size=config.batch_size,
                            **kwargs)

    # Set up error logging and wandb for global tracking
    os.environ["WANDB_SILENT"] = "true"
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()
    wandb.init(config=config,project="SGVAE", 
               entity="brichardson",
               id=wandb_id,
               group=f'keep_s:{config.keep_style},update_prior:{config.update_prior}',
               resume="allow",
               config_include_keys=('action_repetitions','dataset','keep_style',
                                    'update_prior','style_coef','beta_VAE',
                                    'style_dim'))
    wandb.watch(models=[encoder,decoder])
    train_loss_logger = logger.vae_loss_logger(config,'Training')
    val_loss_logger = logger.vae_loss_logger(config,'Validation')

    # make folders if needed, save train/val indices if necessary
    os.makedirs(config.save_path,exist_ok=True)
    if checkpoint is False:
        with open(f'{config.save_path}/split_indices.json', 'w') as ix_file:
            json.dump(indices, ix_file)
    config.save_path = f'{config.save_path}/checkpoints'
    os.makedirs(config.save_path,exist_ok=True)

    # training loop
    start_epoch = checkpoint+1 if checkpoint else 0
    for epoch in range(start_epoch, config.total_epochs):
        train_loss_logger.reset_epoch_loss()
        val_loss_logger.reset_epoch_loss()

        training_set.random_context_sampler()
        validation_set.random_context_sampler()

        print('')
        print('Epoch #' + str(epoch) + '......................................................................')
        trl.train_epoch(config,encoder,decoder,optimizer,
                        train_loader,train_loss_logger,
                        epoch)

        # val every 10 epochs
        if (epoch + 1) % 5 == 0:
            trl.eval(config, encoder, decoder, val_loader, val_loss_logger, 
                     epoch)
        # save checkpoint at end of time or last epoch
        try:
            # limit total amount of time the job can run on the cluster.
            if ((time.perf_counter()-start_time) > config.cluster_timer):
                utils.save_vae_checkpoint(config.save_path,epoch,wandb_id,
                                        encoder,decoder,optimizer)
                print('Exiting, should hold in cluster and resume')
                return 3
        except:
            if (epoch + 1) % 25 == 0:
                utils.save_vae_checkpoint(config.save_path,epoch,wandb_id,
                                        encoder,decoder,optimizer)
                
    utils.save_vae_checkpoint(config.save_path,epoch,wandb_id,
                                        encoder,decoder,optimizer)
    return 0

if __name__ == '__main__':
    config = cmd_parser.parse_vae_config(sys.argv[1:])
    sys.exit(main(config))


