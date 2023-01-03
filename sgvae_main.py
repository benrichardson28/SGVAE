import argparse
import torch
import pdb
from training import training_procedure

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")

parser.add_argument('--initial_learning_rate', type=float, default=0.001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

parser.add_argument('--style_dim', type=int, default=10, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=10, help="dimension of common factor latent space")
#parser.add_argument('--num_classes', type=int, default=10, help="number of classes on which the data set trained")
#parser.add_argument('--class_acc_num', type=int, default=10, help="number of samples from a class to accumulate")

parser.add_argument('--class_coef', type=float, default=1., help="coefficient for class KL")
parser.add_argument('--style_coef', type=float, default=1., help="coefficient for style KL")
parser.add_argument('--beta_vae',type=float, default=.1, help="beta coefficient for beta-VAE")
parser.add_argument('--reduction', type=str, default='mean')
parser.add_argument('--weight_style', type=bool, default=False, help="weigh reconstructions more when they come from real style")

# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder', help="model save for decoder")

parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")

parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=1000, help="flag to indicate the final epoch of training")

parser.add_argument('--in_channels', type=int, default=195)
parser.add_argument('--hidden_dims', type=int, nargs='*', default=[128,128,128,128]) #[128,128,128])
parser.add_argument('--kernels', type=int, nargs='*', default=[6,6,5,4]) #[6,6,6])
parser.add_argument('--strides', type=int, nargs='*', default=[4,4,2,2]) #[3,3,3])
parser.add_argument('--paddings', type=int, nargs='*', default=[1,1,1,0]) #[1,1,2])
parser.add_argument('--beta_nll', type=float, default=0.5)

# parser.add_argument('--hidden_dims', type=int, nargs='*', default=[128,128,128])
# parser.add_argument('--kernels',type=list, default=[6,6,6])
# parser.add_argument('--strides',type=list, default=[3,3,3])
# parser.add_argument('--paddings',type=list, default=[1,1,2])

FLAGS = parser.parse_args()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    training_procedure(FLAGS)
