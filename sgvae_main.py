import sys
import os
import os.path
import cmd_parser
import yaml
import json
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import training as trl
import utils
import losses
import wandb
from datetime import datetime


def main(config):
    if torch.cuda.is_available() and (config.device=='gpu'):
        print('Cuda available; running on gpu.')
        config.device = torch.device('cuda')
    else: 
        print('Cuda unavailable; running on cpu.')
        config.device = torch.device('cpu')

    # models and optimizer
    print('Creating models.')
    encoder,decoder = utils.create_vae_models(config)
    optimizer = utils.create_vae_optimizer(config,encoder,decoder)

    indices = None
    if config.load_checkpoint:
        print('Loading checkpoint.')
        encoder,decoder,optimizer,_,indices = utils.load_vae_checkpoint(
                                                config.load_checkpoint,
                                                config.start_epoch,
                                                encoder,decoder,optimizer)
    encoder.to(config.device)
    decoder.to(config.device)
    
    # datasets
    print('Creating datasets.')
    training_set, validation_set, indices = utils.create_vae_training_datasets(config,indices)

    # dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(training_set,batch_size=config.batch_size,
                        shuffle=True,**kwargs)
    val_loader = DataLoader(validation_set,batch_size=config.batch_size,
                            **kwargs)

    # writer and tracker
    wandb_id = wandb.util.generate_id()
    wandb.init(config=config,project="SGVAE", entity="brichardson",id=wandb_id)

    #config.save_path=writer.logdir
    config.save_path = os.path.join('runs',datetime.now().strftime("%m%d%Y-%H:%M:%S-")+ config.save_path)
    train_loss_logger = losses.vae_loss_logger(config,'Training')
    val_loss_logger = losses.vae_loss_logger(config,'Validation')

    # with open(f'{config.logdir}/../run_tracker.csv', mode='a+',newline='') as tracking_file:
    #     meta_track = csv.DictWriter(tracking_file,fieldnames=['Output_folder',*vars(FLAGS).keys()])
    #     if len(open(f'{FLAGS.logdir}/../run_tracker.csv', 'r').readlines())<=1:
    #         meta_track.writeheader()
    #     hlink = f'=HYPERLINK("/home/richardson/cluster/robot_haptic_perception/{FLAGS.logdir}")'
    #     meta_track.writerow({'Output_folder':hlink,**vars(FLAGS)})
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    with open(f'{config.save_path}/config.yaml', 'w') as conf_file:
        yaml.dump(config, conf_file)
    with open(f'{config.save_path}/split_indices.json', 'w') as ix_file:
        json.dump(indices, ix_file)
    config.save_path = f'{config.save_path}/checkpoints'
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)


    for epoch in range(config.end_epoch - config.start_epoch):
        train_loss_logger.reset_epoch_loss()
        val_loss_logger.reset_epoch_loss()

        training_set.random_context_sampler()
        validation_set.random_context_sampler()

        print('')
        print('Epoch #' + str(epoch) + '......................................................................')
        trl.train_epoch(config,encoder,decoder,optimizer,
                        train_loader,train_loss_logger,
                        epoch)

        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.end_epoch:
            trl.eval(config, encoder, decoder, val_loader, val_loss_logger, 
                     epoch)
            utils.save_vae_checkpoint(config.save_path,epoch,encoder,decoder)


if __name__ == '__main__':
    config = cmd_parser.parse_vae_config(sys.argv[1:])
    main(config)


