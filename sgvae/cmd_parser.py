#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:59:27 2023

@author: richardson
"""

import configargparse
from distutils.util import strtobool

def parse_vae_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of surface-perception learning model'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description)
    parser.add_argument('-c', '--config', required=True,
                        is_config_file=True,
                        help='config file')

    parser.add_argument('--cluster_timer', type=int, default=None, help="for cluster: How long to run before restarting. Only useful for cluser.")
    parser.add_argument('--device',type=str, default='gpu',choices=['gpu','cpu'])
    # path parameters
    parser.add_argument('--data_path', type=str, default="/fast/richardson/robot_grasp_data/")
    parser.add_argument('--save_path', type=str, default="runs/1")

    # training
    parser.add_argument('--total_epochs', type=int, default=10000, help="Max epochs to train.")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")

    # optimizer
    parser.add_argument('--initial_learning_rate', type=float, default=0.001, help="starting learning rate")
    parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

    # architecture
    parser.add_argument('--style_dim', type=int, default=10, help="dimension of varying factor latent space")
    parser.add_argument('--content_dim', type=int, default=10, help="dimension of common factor latent space")
    parser.add_argument('--keep_style',default=True,type=lambda x: bool(strtobool(x)), help="If true, style is save for all future iterations")
    parser.add_argument('--update_prior',default=False,type=lambda x: bool(strtobool(x)), help="If true, content is used as next prior instead of being fed into the network.")
    
    parser.add_argument('--in_channels', type=int, default=195)
    parser.add_argument('--hidden_dims', type=int, nargs='*', default=[128,64,64,32])
    parser.add_argument('--kernels', type=int, nargs='*', default=[6,6,5,4])
    parser.add_argument('--strides', type=int, nargs='*', default=[4,4,2,2])
    parser.add_argument('--paddings', type=int, nargs='*', default=[1,1,1,0])

    # loss
    parser.add_argument('--content_coef', type=float, default=1., help="coefficient for class KL")
    parser.add_argument('--style_coef', type=float, default=1., help="coefficient for style KL")
    parser.add_argument('--beta_VAE',type=float, default=.1, help="beta coefficient for beta-VAE")
    parser.add_argument('--reduction', type=str, default='mean')
    parser.add_argument('--beta_NLL', type=float, default=1., help="beta coefficient for NLL")
    parser.add_argument('--weight_style', type=bool, default=False, help="weigh reconstructions more when they come from real style")

    # dataset type
    parser.add_argument('--dataset', type=str, default='new', choices=['all','orig','new'])
    parser.add_argument('--action_repetitions', type = int, default = 1)
    parser.add_argument('--split_ratio',type=int,default=4,help="1/x: What fraction of data should be reserved for validation.")

    args = parser.parse_args(argv)

    return args

def parse_inference_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of surface-perception learning model'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description)
    parser.add_argument('-c', '--config', required=True,
                        is_config_file=True,
                        help='config file')

    parser.add_argument('--device',type=str, default='gpu',choices=['gpu','cpu'])
    parser.add_argument('--save_path', type=str, default=0)
    parser.add_argument('--vae_model_path', type=str)
    parser.add_argument('--vae_checkpoint', type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--end_epoch', type=int, default=1, help="flag to indicate the final epoch of training")

    parser.add_argument('--initial_learning_rate', type=float, default=0.001)
    parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
    parser.add_argument('--weight_decay', type=float, default=1, help="weight decay for adam")

    parser.add_argument('--repetitions',type=int, default=5, help="How many random random sequences to generate for training")
    parser.add_argument('--test_properties', type=str, nargs='*', default=['height','width','stiffness','mass','contents_binary_label'])
    parser.add_argument('--action_select', type=str, default=None, choices=['squeeze', 'press', 'shake', 'slide'])

    args = parser.parse_args(argv)

    return args
