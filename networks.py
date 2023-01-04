import torch
import torch.nn as nn
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, style_dim, class_dim,**kwargs):
        super(Encoder, self).__init__()

        in_channels = kwargs['in_channels']
        modules = []
        cos = 400
        for k,s,p,h_dim in zip(kwargs['kernels'],kwargs['strides'],kwargs['paddings'],kwargs['hidden_dims']):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size = k, stride = s, padding = p),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            self.inner_size = h_dim
            cos = 1+int((cos-k+2*p)/s)
        self.conv = nn.Sequential(*modules)
        self.cos = cos
        in_features = kwargs['hidden_dims'][-1]*cos + 2*class_dim
        mlp_sz = 100
        self.lin1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=mlp_sz, bias=True),
                                  nn.LeakyReLU())

        # style
        self.style_mu = nn.Linear(in_features=mlp_sz, out_features=style_dim, bias=True)
        self.style_logvar = nn.Linear(in_features=mlp_sz, out_features=style_dim, bias=True)

        # class
        self.class_mu = nn.Linear(in_features=mlp_sz, out_features=class_dim, bias=True)
        self.class_logvar = nn.Linear(in_features=mlp_sz, out_features=class_dim, bias=True)

    def forward(self, x, context=None):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x,context],dim=1)
        x = self.lin1(x)

        style_latent_space_mu = self.style_mu(x)
        style_latent_space_logvar = self.style_logvar(x)

        class_latent_space_mu = self.class_mu(x)
        class_latent_space_logvar = self.class_logvar(x)

        return style_latent_space_mu, style_latent_space_logvar, \
                class_latent_space_mu, class_latent_space_logvar


class Decoder(nn.Module):
    def __init__(self, style_dim, class_dim, **kwargs):
        super(Decoder, self).__init__()

        modules = []
        hidden_dims = kwargs['hidden_dims'].copy()
        kernels = kwargs['kernels'].copy()
        strides = kwargs['strides'].copy()
        pads = kwargs['paddings'].copy()
        self.cos = kwargs['cos']
        # self.decoder_input = nn.Linear(style_dim + class_dim + action_dim, hidden_dims[-1] * 4)
        self.decoder_input = nn.Sequential(
            nn.Linear(style_dim + class_dim + 4, 100),
            nn.LeakyReLU(),
            nn.Linear(100, hidden_dims[-1] * self.cos))
        hidden_dims.reverse()
        kernels.reverse()
        strides.reverse()
        pads.reverse()
        self.inner_size = hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=kernels[i],
                                       stride = strides[i],
                                       padding=pads[i],
                                       output_padding=0),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        def create_last_layer():
            final_layer = nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[-1],
                                       hidden_dims[-1],
                                       kernel_size=kernels[-1],
                                       stride=strides[-1],
                                       padding=pads[-1],
                                       output_padding=pads[-1]),
                    nn.BatchNorm1d(hidden_dims[-1]),
                    nn.LeakyReLU(),
                    nn.Conv1d(hidden_dims[-1], out_channels= kwargs['out_channels'],
                              kernel_size = kernels[-1], padding= 2*pads[-1]),
                    nn.Tanh())
            return final_layer
        self.final_layer_mu = create_last_layer()
        self.final_layer_var = create_last_layer()

    def forward(self, style_latent_space, class_latent_space, action):
        x = torch.cat((style_latent_space,class_latent_space, action), dim=-1) # was dim=2 when crops are kept together
        #x = torch.cat((style_latent_space,class_latent_space), dim=-1)
        ds = x.shape
        x = self.decoder_input(x)
        x = x.view(-1, self.inner_size, self.cos)
        x = self.decoder(x)
        mu = self.final_layer_mu(x)
        logvar = self.final_layer_var(x)
        os = mu.shape
        #return mu,logvar
        return mu.view(ds[0],ds[1],os[-2],os[-1]), logvar.view(ds[0],ds[1],os[-2],os[-1])

class Property_model(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Property_model, self).__init__()

        self.fc_model = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=50, bias=True),
            # ('fc_1_bn', nn.BatchNorm1d(num_features=10)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # ('fc_2', nn.Linear(in_features=30, out_features=256, bias=True)),
            # ('fc_2_bn', nn.BatchNorm1d(num_features=256)),
            # ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            nn.Linear(in_features=50, out_features=num_classes, bias=True)
        )
        self.num_classes = num_classes

    def forward(self, z):
        x = self.fc_model(z)
        if self.num_classes==1:
            x=x.flatten()
        return x
