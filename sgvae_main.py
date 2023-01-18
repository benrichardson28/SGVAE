import sys
import cmd_parser
import torch
from training import training_procedure
import utils


def main(config):
    encoder,decoder = utils.setup_vae_models(config)
    training_procedure(config)

    

if __name__ == '__main__':
    config = cmd_parser.parse_vae_config(sys.argv[1:])
    main(config)

