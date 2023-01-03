import os
import yaml
from argparse import Namespace
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms as transforms
import pdb
import numpy as np
from torchvision import datasets
from networks import Encoder,Decoder,baseVAE_encoder,baseVAE_decoder

def setup_models(run_folder):
    config_file = f'{run_folder}/config.yaml'
    with open(config_file, 'r') as file:
        FLAGS = yaml.load(file,Loader=yaml.UnsafeLoader)
    if type(FLAGS) is dict:
        FLAGS = Namespace(**FLAGS)
    if 'logp_weight' not in FLAGS: setattr(FLAGS,'logp_weight',0)
    # if FLAGS.hidden_dims[0]!=128:
    #     FLAGS.hidden_dims.reverse()
    #     FLAGS.kernels.reverse()
    #     FLAGS.paddings.reverse()
    #     FLAGS.strides.reverse()

    """
    model definitions
    """


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'style_dim' not in FLAGS:
        encoder = baseVAE_encoder(latent_dim=FLAGS.latent_dim,
                                  in_channels = FLAGS.in_channels,
                                  hidden_dims = FLAGS.hidden_dims,
                                  kernels = FLAGS.kernels,
                                  strides = FLAGS.strides,
                                  paddings = FLAGS.paddings).to(FLAGS.device)
    #encoder.apply(weights_init)
        decoder = baseVAE_decoder(latent_dim=FLAGS.latent_dim,
                                  hidden_dims = FLAGS.hidden_dims,
                                  out_channels = FLAGS.in_channels,
                                  kernels = FLAGS.kernels,
                                  strides = FLAGS.strides,
                                  paddings = FLAGS.paddings,
                                  cos = encoder.cos).to(FLAGS.device)
    else:
        encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim,
                          in_channels = FLAGS.in_channels,
                          hidden_dims = FLAGS.hidden_dims,
                          kernels = FLAGS.kernels,
                          strides = FLAGS.strides,
                          paddings = FLAGS.paddings).to(FLAGS.device)
        #encoder.apply(weights_init)
        decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim,
                          hidden_dims = FLAGS.hidden_dims,
                          out_channels = FLAGS.in_channels,
                          kernels = FLAGS.kernels,
                          strides = FLAGS.strides,
                          paddings = FLAGS.paddings,
                          cos = encoder.cos).to(FLAGS.device)

    folder = f'{run_folder}/checkpoints_{FLAGS.batch_size}'

    #monitor = torch.load(os.path.join(folder, 'monitor_e%d'%(FLAGS.end_epoch -1)))

    #monitor[0,0] =  - np.inf
    #best_elbo = monitor[checks,0].argmax()

    #DEBUG
    best_elbo=-1
    mx=0
    for rn in os.listdir(folder):
        mx = max(mx,int(''.join(filter(lambda i: i.isdigit(), rn[-5:]))))
    #mx=199
    #print(checks[best_elbo],monitor[checks[best_elbo],0],monitor[checks,0].max(), best_elbo)
    FLAGS.encoder_save = folder + '/encoder_e%d' % mx #checks[best_elbo]
    FLAGS.decoder_save = folder + '/decoder_e%d' % mx #checks[best_elbo]

    #FLAGS.batch_size = 256    #the first use is to load the file
    encoder.load_state_dict(torch.load(FLAGS.encoder_save, map_location=lambda storage, loc: storage))
    decoder.load_state_dict(torch.load(FLAGS.decoder_save, map_location=lambda storage, loc: storage))

    encoder.eval()
    decoder.eval()
    encoder.to(device)
    decoder.to(device)

    return encoder,decoder,FLAGS,device

def cNs_init(batch_size,act_num,c_size,s_size,device):
    context = torch.cat([torch.zeros(batch_size,c_size),
                         torch.ones(batch_size,c_size)/2.0],dim=1).to(device)

    style_mu = torch.zeros(batch_size,act_num,s_size).to(device)
    style_logvar = 0.5*torch.ones(batch_size,act_num,s_size).to(device)

    return context, style_mu, style_logvar

def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = torch.zeros_like(std).normal_()
        return eps.mul(std).add_(mu)
    else:
        return mu

def group_wise_reparameterize(training, mu, logvar, labels_batch,
                            list_groups_labels, sizes_group, cuda):
    eps_dict = {}
    batch_size = labels_batch.size(0)
    # generate only 1 eps value per group label
    for i, g in enumerate(list_groups_labels):
        if cuda:
            eps_dict[g] = torch.cuda.FloatTensor(1, logvar.size(1)).normal_()
        else:
            eps_dict[g] = torch.FloatTensor(1, logvar.size(1)).normal_()

    if training:
        std = logvar.mul(0.5).exp_()
    else:
        std =torch.zeros_like(logvar)

    content_samples = []
    indexes = []
    sizes = []
    # multiply std by correct eps and add mu
    for i, g in enumerate(list_groups_labels):
        samples_group = labels_batch.eq(g).nonzero().squeeze()
        size_group = samples_group.numel()
        assert size_group == sizes_group[i]
        if size_group > 0:

            reparametrized = std[i][None,:] * eps_dict[g] + mu[i][None,:]
            group_content_sample = reparametrized.repeat((size_group,1))
            content_samples.append(group_content_sample)
            if size_group == 1:
                samples_group = samples_group[None]
            indexes.append(samples_group)
            size_group = torch.ones(size_group) * size_group
            sizes.append(size_group)

    content_samples = torch.cat(content_samples,dim=0)
    indexes = torch.cat(indexes)
    sizes = torch.cat(sizes)

    return content_samples, indexes, sizes

def group_wise_reparameterize_each(training, mu, logvar, labels_batch,
                            list_groups_labels, sizes_group, cuda):
    eps_dict = {}
    batch_size = labels_batch.size(0)

    if training:
        std = logvar.mul(0.5).exp_()
    else:
        std =torch.zeros_like(logvar)

    content_samples = []
    indexes = []
    sizes = []


    # multiply std by correct eps and add mu
    for i, g in enumerate(list_groups_labels):
        samples_group = labels_batch.eq(g).nonzero().squeeze()
        size_group = samples_group.nelement()
        #always get annoying size error when nelements is 0 or 1
        try:
            if len(samples_group.size()) > 0:
                size_group = samples_group.size(0)
        except:
            pdb.set_trace()
        #assert size_group == sizes_group[i]
        if size_group > 0:
            if cuda:
                eps = torch.cuda.FloatTensor(size_group, std.size(1)).normal_()
            else:
                eps = torch.FloatTensor(size_group, std.size(1)).normal_()
            group_content_sample = std[i][None,:] * eps + mu[i][None,:]
            content_samples.append(group_content_sample)
            if size_group == 1:
                samples_group = samples_group[None]
            indexes.append(samples_group)
            size_group = torch.ones(size_group) * size_group
            sizes.append(size_group)
    content_samples = torch.cat(content_samples,dim=0)
    indexes = torch.cat(indexes)
    sizes = torch.cat(sizes)

    return content_samples, indexes, sizes

def weights_init(layer):
    r"""Apparently in Chainer Lecun normal initialisation was the default one
    """
    if isinstance(layer, nn.Linear):
        lecun_normal_(layer.bias)
        lecun_normal_(layer.weight)

def lecun_normal_(tensor, gain=1):

    import math
    r"""Adapted from https://pytorch.org/docs/0.4.1/_modules/torch/nn/init.html#xavier_normal_
    """
    dimensions = tensor.size()
    if len(dimensions) == 1:  # bias
        fan_in = tensor.size(0)
    elif len(dimensions) == 2:  # Linear
        fan_in = tensor.size(1)
    else:
        num_input_fmaps = tensor.size(1)
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size

    std = gain * math.sqrt(1.0 / (fan_in))
    with torch.no_grad():
        return tensor.normal_(0, std)

def imshow_grid(images, shape=[2, 8], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    if save:
        plt.savefig('reconstructed_images/' + str(name) + '.png')
        plt.clf()
    else:
        plt.show()
