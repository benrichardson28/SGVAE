import torch
import torch.nn as nn

class Encoder(nn.Module):
    """ Encoder module for embedding tactile data in latent space.
    
    :param int style_dim: Size of style latent space.
    :param int content_dim: Size of content latent space.
    :param int data_len: Length of data (used for computing final conv output size)
    
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * in_channels (int) -- Number of input channels
        * kernels (list) -- Convolution kernel sizes for each layer.
        * strides (list) -- Stride lengths for kernels at each layer.
        * paddings (list) -- Padding to apply to input for each layer.
        * hidden_dims (list) -- Number of kernels to use at each layer.
    """
    def __init__(self, style_dim, content_dim,data_len=400,**kwargs):
        super(Encoder, self).__init__()

        in_channels = kwargs['in_channels']
        modules = []
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
            data_len = 1+int((data_len-k+2*p)/s)
        self.conv = nn.Sequential(*modules)
        self.data_len = data_len
        in_features = kwargs['hidden_dims'][-1]*data_len + 2*content_dim
        #if not kwargs['remove_context']: in_features += 2*content_dim

        mlp_sz = 100
        self.lin1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=mlp_sz, bias=True),
                                  nn.LeakyReLU())

        # style
        self.style_mu = nn.Linear(in_features=mlp_sz, out_features=style_dim, bias=True)
        self.style_logvar = nn.Linear(in_features=mlp_sz, out_features=style_dim, bias=True)

        # class
        self.content_mu = nn.Linear(in_features=mlp_sz, out_features=content_dim, bias=True)
        self.content_logvar = nn.Linear(in_features=mlp_sz, out_features=content_dim, bias=True)

    def forward(self, x, context=None):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        if context is not None:
            x = torch.cat([x,context],dim=1)
        x = self.lin1(x)

        style_latent_space_mu = self.style_mu(x)
        style_latent_space_logvar = self.style_logvar(x)

        content_latent_space_mu = self.content_mu(x)
        content_latent_space_logvar = self.content_logvar(x)

        return style_latent_space_mu, style_latent_space_logvar, \
                content_latent_space_mu, content_latent_space_logvar


class Decoder(nn.Module):
    """ Decoder module for reconstructing tactile data from latent representation.
    
    :param int style_dim: Size of style latent space.
    :param int content_dim: Size of content latent space.
    
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * in_channels (int) -- Number of input channels
        * kernels (list) -- Convolution kernel sizes for each layer.
        * strides (list) -- Stride lengths for kernels at each layer.
        * paddings (list) -- Padding to apply to input for each layer.
        * hidden_dims (list) -- Number of kernels to use at each layer.
    """
    def __init__(self, style_dim, content_dim, data_len, **kwargs):
        super(Decoder, self).__init__()

        modules = []
        hidden_dims = kwargs['hidden_dims'].copy()
        kernels = kwargs['kernels'].copy()
        strides = kwargs['strides'].copy()
        pads = kwargs['paddings'].copy()
        self.data_len = kwargs['data_len']
        # self.decoder_input = nn.Linear(style_dim + content_dim + action_dim, hidden_dims[-1] * 4)
        self.decoder_input = nn.Sequential(
            nn.Linear(style_dim + content_dim + 4, 100),
            nn.LeakyReLU(),
            nn.Linear(100, hidden_dims[-1] * self.data_len))
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

    def forward(self, style_latent_space, content_latent_space, action):
        x = torch.cat((style_latent_space,content_latent_space, action), dim=-1) # was dim=2 when crops are kept together
        #x = torch.cat((style_latent_space,content_latent_space), dim=-1)
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
            nn.Linear(in_features=z_dim, out_features=80, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=80, out_features=80, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=80, out_features=40, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=40, out_features=num_classes, bias=True)
        )
        self.num_classes = num_classes

    def forward(self, z):
        x = self.fc_model(z)
        if self.num_classes==1:
            x=x.flatten()
        return x
