
import math
import torch
import sgvae.utils as utils

def loss(config,cm,clv,sm,slv,mu_x,logvar_x,X,prior=None,style_weights=None):
    
    # KL for multi-var gaussians. 
    # For standard VAE, use KL(estimate || true posterior)
    def multi_var_KL(m1,lv1,m2,lv2):
        ### EQ: 1/2[ log(det VAR_2/det VAR_1) - n + tr{inv(VAR_2) * VAR_1} 
        ###          + (MEAN_2 - MEAN_1)^T * inv(VAR_2) * (MEAN_2 - MEAN_1)]  
        kl = 0.5*( (lv2-lv1+(lv1-lv2).exp() \
                    +(m2-m1).pow(2)/lv2.exp()).sum(dim=1)  \
                    - m1.shape[-1] )
        if config.reduction=="sum": return kl.sum()
        elif config.reduction=="mean": return kl.mean()
        return kl
    
    # define standard normal prior
    prior_mu = torch.zeros_like(cm)
    prior_logvar = torch.zeros_like(clv)
    # use
    style_KL = multi_var_KL(sm,slv,prior_mu,prior_logvar)
    
    if prior is not None:
        prior_mu,prior_logvar = prior
    content_KL = multi_var_KL(cm,clv,prior_mu,prior_logvar)

    # if config.reduction=='sum':
    #     style_kl_divergence_loss = 0.5 * ( - 1 - slv + sm.pow(2) + slv.exp()).sum()
    #     content_kl_divergence_loss = 0.5 * ( - 1 - clv + cm.pow(2) + clv.exp()).sum()
    # elif config.reduction=='mean':
    #     style_kl_divergence_loss = 0.5 * ( - 1 - slv + sm.pow(2) + slv.exp()).sum(dim=1).mean()
    #     content_kl_divergence_loss = 0.5 * ( - 1 - clv + cm.pow(2) + clv.exp()).sum(dim=1).mean()
    #BUILD IN THE KL BETWEEN the current content and the previous content (context)
    
    ####################################################################


    #### gaussian_beta_log_likelihood_loss(pred, target, beta=1):
    scale_x = (torch.exp(logvar_x) + 1e-12)#**0.5
    mean, var = torch.squeeze(mu_x,1),torch.squeeze(scale_x)
    logl = -0.5 * ((X - mean) ** 2 / var + torch.log(var) + math.log(2 * math.pi))
    weight = var.detach() ** config.beta_NLL

    if config.reduction=='sum':
        logp_batch = torch.sum(logl * weight, axis=-1).sum(-1)
        reconstruction_proba = logp_batch.sum()
    elif config.reduction=='mean':
        logp_batch = torch.sum(logl * weight, axis=-1).sum(-1)
        if style_weights is not None:
            logp_batch *= style_weights
        reconstruction_proba = logp_batch.mean(-1).mean()
  
    total_KL = config.style_coef*style_KL + config.content_coef*content_KL
    elbo = (reconstruction_proba - config.beta_VAE * total_KL)

    return elbo, reconstruction_proba, style_KL, content_KL

def process(config, X, action_batch, encoder, decoder, loss_logger):
    context, style_mu, style_logvar = utils.cNs_init(config,X.shape[0])
    X = X.to(config.device)
    action_batch=action_batch.to(config.device)

    total_elbo = 0
    # context loop

    #pdb.set_trace()
    for cs in range(X.size(1)):
        #pass in first sample -> get content and style
        sm,slv,cm,clv,mu_x,logvar_x = single_pass(config,X,action_batch,cs,
                                                  context,style_mu,style_logvar,
                                                  encoder,decoder)

        style_weights = None
        if config.weight_style:
            style_weights = torch.ones(X.size(0),X.size(1))
            style_weights = style_weights.to(config.device)
            style_weights[:,:-1-cs] *= 0.5

        prior = None
        if config.update_prior:
            prior = torch.tensor_split(context,2,dim=1)

        elbo, mle, kl_style, \
            kl_content = loss(config,cm,clv,sm,slv,mu_x,logvar_x,X,prior,style_weights)
        total_elbo += elbo*1.
        loss_logger.update_epoch_loss(elbo,mle,kl_content,kl_style,cs)

        # prepare latents for next pass: create context
        context = torch.cat([cm,clv],dim=1)

    return total_elbo / (config.action_repetitions * EXPLORATORY_PROCEDURE_NUM)

def single_pass(config,X,action_batch,cs,
                context,style_mu,style_logvar,
                encoder,decoder=None,training=True):
    #style vars should be updated outside the function
    #UPDATE SO THAT THE STYLE IS NOT SAVED AFTER IT IS USED ONCE


    ####################################################################

    sm,slv,cm,clv = encoder(X[:,-1-cs],context)
    #add on other styles, concat content
    if config.keep_style:
        style_mu[:,-1-cs] = sm.detach().clone()
        style_logvar[:,-1-cs] = slv.detach().clone()
    content_mu = cm.unsqueeze(1).repeat([1,X.size(1),1])
    content_logvar = clv.unsqueeze(1).repeat([1,X.size(1),1])

    if decoder is None:
        return sm,slv,cm,clv,None,None
    
    #reparam
    content_latent_embeddings = utils.reparameterize(training=training, mu=content_mu, logvar=content_logvar)   #batch x 4 x 10
    single_style_latent = utils.reparameterize(training=training, mu=sm, logvar=slv)   #batch x 10
    style_latent_embeddings = utils.reparameterize(training=training, mu=style_mu, logvar=style_logvar)

    style_latent_embeddings[:,-1-cs] = single_style_latent

    #reconstruct with action
    mu_x, logvar_x = decoder(style_latent_embeddings, content_latent_embeddings, action_batch)

    return sm,slv,cm,clv,mu_x,logvar_x


def eval(config, encoder, decoder, loader, logger, epoch):

    with torch.no_grad():
        for it, (data_batch, action_batch, _) in enumerate(loader):

            _ = process(config, data_batch, action_batch, 
                        encoder, decoder, logger)

    logger.finalize_epoch_loss(it+1)  #type: ignore
    logger.logwandb(epoch)

def train_epoch(config,encoder,decoder,optimizer,
                train_loader,train_logger,
                epoch):
    it = 0
    for it, (data_batch, action_batch, _) in enumerate(train_loader):
        # set zero_grad for the optimizer
        optimizer.zero_grad()
        X = data_batch.to(config.device).detach().clone()

        elbo = process(config, X, action_batch,
                        encoder, decoder, train_logger)
        (-elbo).backward()
        optimizer.step()

        print(f'\r{it}/{len(train_loader)}',end='')

    train_logger.finalize_epoch_loss(it+1)   
    train_logger.print_losses()
    if (epoch + 1) % 10:
        train_logger.logwandb(epoch)