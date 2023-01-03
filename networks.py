
import torch
import torch.nn as nn
import pdb

#modify to add different quadrant information
class ConvEnc(nn.Module):
    def __init__(self,channels,layer_dims,kernels,strides,paddings,**kwargs):
        super(ConvEnc, self).__init__()
        in_channels = channels
        modules = []
        for h_dim,k,s,p in zip(layer_dims,kernels,strides,paddings):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size = k, stride = s, padding = p),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            self.inner_size = h_dim
        self.conv = nn.Sequential(*modules)

    def forward(self,x):
        return self.conv(x)
        
        
class mlvaeEncoder(nn.Module):
    def __init__(self, style_dim, content_dim,
                 mlp_sz, channels, layer_dims, #mlp=100
                 kernels, strides, paddings,
                 **kwargs):
        super(Encoder, self).__init__()
        self.conv = ConvEnc(channels,layer_dims,kernels,strides,paddings)
        
        in_features = layer_dims[-1]*self.conv.out_shp + 2*content_dim
        self.lin1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=mlp_sz, bias=True),
                                  nn.LeakyReLU())
        # style
        self.style_mu = nn.Linear(in_features=mlp_sz, out_features=style_dim, bias=True)
        self.style_logvar = nn.Linear(in_features=mlp_sz, out_features=style_dim, bias=True)

        # class
        self.content_mu = nn.Linear(in_features=mlp_sz, out_features=content_dim, bias=True)
        self.content_logvar = nn.Linear(in_features=mlp_sz, out_features=content_dim, bias=True)

        #action
        # self.action_mu = nn.Linear(in_features=in_features, out_features=action_dim, bias=True)
        # self.action_logvar = nn.Linear(in_features=in_features, out_features=action_dim, bias=True)

    def forward(self, x, context=None):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x,context],dim=1)
        x = self.lin1(x)

        content_latent_space_mu = self.content_mu(x)
        content_latent_space_logvar = self.content_logvar(x)
                
        style_latent_space_mu = self.style_mu(x)
        style_latent_space_logvar = self.style_logvar(x)

        return content_latent_space_mu, content_latent_space_logvar, \
            style_latent_space_mu, style_latent_space_logvar

class mlvaeDecoder(nn.Module):
    def __init__(self, input_dim, mlp_sz, layer_dims,
                 kernels, strides, paddings, **kwargs):
        super(Decoder, self).__init__()
        #for ML input_dim  = class_dim+style_dim+action_dim
        modules = []
        
        #compute shape of conv_output
        conv_out_shape = 400
        for k,s,p in zip(kernels,strides,paddings):
            conv_out_shape = 1+int((conv_out_shape-k+2*p)/s)
        
        # self.decoder_input = nn.Linear(style_dim + class_dim + action_dim, hidden_dims[-1] * 4)
        self.decoder_input = nn.Sequential(
            nn.Linear(input_dim , mlp_sz),
            nn.LeakyReLU(),
            nn.Linear(mlp_sz, layer_dims[-1] * conv_out_shape))
        
        layer_dims = layer_dims[::-1]
        kernels = kernels[::-1]
        strides = strides[::-1]
        pads = paddings[::-1]
        self.inner_size = layer_dims[0]
  
        for i in range(len(layer_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(layer_dims[i],
                                       layer_dims[i + 1],
                                       kernel_size=kernels[i],
                                       stride = strides[i],
                                       padding=pads[i],
                                       output_padding=0),
                    nn.BatchNorm1d(layer_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        def create_last_layer():
            final_layer = nn.Sequential(
                    nn.ConvTranspose1d(layer_dims[-1],
                                       layer_dims[-1],
                                       kernel_size=kernels[-1],
                                       stride=strides[-1],
                                       padding=pads[-1],
                                       output_padding=pads[-1]),
                    nn.BatchNorm1d(layer_dims[-1]),
                    nn.LeakyReLU(),
                    nn.Conv1d(layer_dims[-1], out_channels= kwargs['out_channels'],
                              kernel_size = kernels[-1], padding= 2*pads[-1]),
                    nn.Tanh())
            return final_layer
        self.final_layer_mu = create_last_layer()
        self.final_layer_var = create_last_layer()

    def forward(self, latent_vecs):
        # ML-VAE:  latent_vecs = torch.cat((content_latent,style_latent,action_latent),dim=-1)
        #x = torch.cat((latent_dims, action), dim=-1) # was dim=2 when crops are kept together
        #x = torch.cat((style_latent_space,class_latent_space), dim=-1)
        ds = latent_vecs.shape
        x = self.decoder_input(latent_vecs)
        x = x.view(-1, self.inner_size, self.cos)
        x = self.decoder(x)
        mu = self.final_layer_mu(x)
        logvar = self.final_layer_var(x)
        os = mu.shape
        #return mu,logvar
        return mu.view(ds[0],ds[1],os[-2],os[-1]), logvar.view(ds[0],ds[1],os[-2],os[-1])

class Baseline_Model(ConvEnc):
    def __init__(self, output_dim, mlp_sz,**enc_args):
        super(Baseline_Model, self).__init__(enc_args['channels'],
                                             enc_args['layer_dims'],
                                             enc_args['kernels'],
                                             enc_args['strides'],
                                             enc_args['paddings'])
    
        in_features = enc_args['layer_dims'][-1]*self.conv.out_shp
        self.mlp = nn.Sequential(nn.Linear(in_features=in_features,
                                           out_features=mlp_sz, bias=True),
                                 nn.LeakyReLU(),
                                 nn.Linear(in_features=mlp_sz,
                                           output_features=output_dim, bias=True))
        self.num_classes = output_dim
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        if self.num_classes==1:
            x = torch.flatten(x)
        return x

class Property_model(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_classes):
        super(Property_model, self).__init__()

        self.fc_model = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=True))
        self.num_classes = num_classes

    def forward(self, z):
        x = self.fc_model(z)
        if self.num_classes==1:
            x = torch.flatten(x)
        return x
    
    
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

class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):#, hidden_dims):
        super(Classifier, self).__init__()
        hidden_dims = 50
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=hidden_dims, bias=True),
            # ('fc_1_bn', nn.BatchNorm1d(num_features=10)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # ('fc_2', nn.Linear(in_features=30, out_features=256, bias=True)),
            # ('fc_2_bn', nn.BatchNorm1d(num_features=256)),
            # ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            nn.Linear(in_features=hidden_dims, out_features=num_classes, bias=True)
        )

    def forward(self, z):
        x = self.fc_model(z)

        return x


class baseVAE_encoder(nn.Module):
    def __init__(self, latent_dim,**kwargs):
        super(baseVAE_encoder, self).__init__()

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
        in_features = kwargs['hidden_dims'][-1]*cos + 2*latent_dim
        mlp_sz = 100
        self.lin1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=mlp_sz, bias=True),
                                  nn.LeakyReLU())

        # just a single latent space
        self.mu = nn.Linear(in_features=mlp_sz, out_features=latent_dim, bias=True)
        self.logvar = nn.Linear(in_features=mlp_sz, out_features=latent_dim, bias=True)

    def forward(self, x, context=None):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x,context],dim=1)
        x = self.lin1(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu,logvar


class baseVAE_decoder(nn.Module):
    def __init__(self, latent_dim, **kwargs):
        super(baseVAE_decoder, self).__init__()

        modules = []
        hidden_dims = kwargs['hidden_dims'].copy()
        kernels = kwargs['kernels'].copy()
        strides = kwargs['strides'].copy()
        pads = kwargs['paddings'].copy()
        self.cos = kwargs['cos']
        # self.decoder_input = nn.Linear(style_dim + class_dim + action_dim, hidden_dims[-1] * 4)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + 4, 100),
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

    def forward(self, latent_space, action):
        x = torch.cat((latent_space, action), dim=-1) # was dim=2 when crops are kept together   
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

