import os
import sys
import yaml
import numpy as np
from itertools import cycle
from argparse import Namespace

import torch
import random
import pickle
from torchvision import datasets
import torchvision
from torch.autograd import Variable
from ball_datasets import tactile_explorations
from utils import reparameterize

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import training

from mpl_toolkits.axes_grid1 import ImageGrid
import plotting
import pdb

from property_maps import property_df
from utils import setup_models,cNs_init
from classification_accuracy import gen_latent_dataset

#%%
ds_train = tactile_explorations(train=True,dataset='new')
ds_test = tactile_explorations(train=False,dataset='new')
ds_train.set_transform()
ds_test.set_transform(ds_train.get_transform())
#del(ds_train)

#%%
def select_samples(ds,device,total_per_object=1):
    sampled_classes = ds.df['object'].unique().tolist()
    sampled_classes.sort()

    ##%%  random crops
    sample_data=[]
    sample_crops=[]
    sample_targs=[]
    for grp in sampled_classes:
        for _ in range(total_per_object):
            ix = random.SystemRandom().choice(ds.df[ds.df['object']==grp].index)
            # for random
            d,c,t = ds[ix]
            sample_data.append(d)
            sample_crops.append(c)
            sample_targs.append(t)
    X = torch.stack(sample_data).to(device)
    sample_crops = torch.stack(sample_crops).to(device)
    sample_targs = torch.tensor(sample_targs).to(device)
    return X, sample_crops, sample_targs

def content_test(FLAGS,context,X,smp_ix,axis,encoder):
    cnt_ten = torch.cat([context,context])
    mu = context[0,2].item()
    st = torch.exp(context[0,12])**0.5

    width = 5
    rng = np.arange(-width*st,width*st,width*2*st/20)
    new_mu = np.zeros(20)
    new_val = np.zeros(20)

    for i,val in enumerate(rng):
        context[0,2] = mu + val
        #cnt_ten[0]=context
        _,_,cm,clv=encoder(X,context)
        #cnt_ten[1,:10]=cm
        #cnt_ten[1,10:]=clv

        new_mu[i]=cm[0,2].item()
        new_val[i]=torch.exp(clv[0,2]).item()**0.5

        #plotting.paolas_prop(cnt_ten[:,:10].T,torch.exp(cnt_ten[:,10:].T)**0.5,FLAGS.logdir,'content shift',i,smp_ix)
    #plt.figure()
    axis.plot(rng,new_mu)
    axis.plot(rng,new_mu+new_val,'orange',rng,new_mu-new_val,'orange')
    #axis.set_aspect('equal')


#%%

def latent_generate(run_folder,ds,cnt,make_fold=True):
    encoder,decoder,FLAGS,device = setup_models(run_folder)
    if make_fold:
        FLAGS.logdir = f'{run_folder}/latent_plots'
        if not os.path.exists(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)

    ds.random_context_sampler()
    X, sample_crops, sample_targs = select_samples(ds,device,cnt)

    with torch.no_grad():
        context_ar=[]
        style_ar=[]
        context,style_mu,style_logvar = cNs_init(X.size(0),X.size(1),
                                                 FLAGS.class_dim,FLAGS.style_dim,device)
        for i in range(4):
            sm,slv,cm,clv,mu_x,lv_x=training.single_pass(X, sample_crops, i, context,
                                                          style_mu, style_logvar,
                                                          encoder, decoder, training=False)
            print(training.loss(FLAGS,cm,clv,sm,slv,mu_x,lv_x,X))
            context = torch.cat([cm,clv],dim=1)
            style = torch.cat([sm,slv],dim=1)
            context_ar.append(context)
            style_ar.append(style)
        cnt_ten = torch.stack(context_ar,dim=1)
        style_ten = torch.stack(style_ar,dim=1)

    return cnt_ten.cpu(),sample_crops.cpu(),sample_targs.cpu(),style_ten.cpu()


#%% ------- Plot latent distributions
#path = os.path.join(os.environ['HOME'],'cluster/robot_haptic_perception/run3')
dire = os.path.join(os.environ['HOME'],'cluster/robot_haptic_perception')
#file = 'run1/Jun06_16-01-25_g09817'
file = 'runs/Aug26_19-42-33_LABKJK12'
f_id = os.path.join(dire,file)
f_id = 'runs/Aug26_19-42-33_LABKJK12'
#listfiles = os.listdir(path)
from sklearn.manifold import TSNE
#for f_id in ['/home/richardson/cluster/robot_haptic_perception/runs/Aug26_19-42-33_LABKJK12',
#'/home/richardson/cluster/robot_haptic_perception/runs/Jun12_18-17-16_g08938',
#'/home/richardson/cluster/robot_haptic_perception/runs/Jun12_18-17-17_g08635',
#'/home/richardson/cluster/robot_haptic_perception/runs/Jun12_18-17-20_g04436',
#'/home/richardson/cluster/robot_haptic_perception/runs/Jun12_18-17-21_g02939',
#]:
acts=ds_test.df['action'].unique()
cnt=10
last=True
#for f_id in listfiles:
#    if (f_id[:3]!='Jun'): continue


    # do this one in run 1 Jun06_16-01-25_g09817  <------ DO NEXT
#if 'Jun07_23-26-32_g09334' not in f_id: continue
#contents, rev_actions, targets = latent_generate(os.path.join(path,f_id),ds_test,cnt)
contents_tr, rev_act_tr,targ_tr,_ = latent_generate(f_id,ds_train,cnt)
contents, rev_actions, targets, _ = latent_generate(f_id,ds_test,cnt)
contents = np.concatenate([contents_tr,contents])
rev_actions = np.concatenate([rev_act_tr,rev_actions])
targets = np.concatenate([targ_tr,targets])

# pdb.set_trace()
#perform dimensionality reduction on contents
if contents.shape[2]>2:
    new_contents=[]
    for i in range(contents.shape[1]):
        new_contents.append(TSNE(n_components=2, learning_rate='auto',
                                 init='random').fit_transform(contents[:,i]))
    contents = np.stack(new_contents,axis=1)

action_one_hot = torch.flip(torch.tensor(rev_actions),[1]).cpu()
prop_map = property_df()
#%%
act_label_arr = acts[torch.where(action_one_hot==1)[2]].reshape([-1,contents.shape[1]])
for col in ['pr_size','sq_size','stiffness','mass','contents_binary_label']:
    if (col=='ball_id') or (col=='ball_name'): continue
    prop_labels = prop_map.loc[targets,col].values
    discrete=False
    if 'contents' in col:
        discrete=True

    plotting.plot_content_points_by_property(contents,act_label_arr,prop_labels,col,cnt,discrete,last)

#%%
def plot_everything(run_folder,ds_test):
    #run_folder = os.path.join(os.environ['HOME'],'cluster',f_id)

    encoder,decoder,FLAGS,device = setup_models(run_folder)

    FLAGS.logdir = f'{run_folder}/eval_img'
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)


    ds_test.random_context_sampler()


    X, sample_crops, sample_targs = select_samples(ds_test,device)
    #
    #reorder = np.random.permutation(4)
    #X = X[:,reorder]
    #sample_crops = sample_crops[:,reorder]
    #reorder = np.array([0,1,2,3])
    # with torch.no_grad():
    #     #With updated context
    #     plot_params = {'plot':True, 'samples':10*total_per_dig, 'rows':2*total_per_dig,
    #                    'epoch':'Context_Style', 'iter':'Plot', 'grp':None}   #larger means later in process
    #     context,style_mu,style_logvar = cNs_init(X.size(0),X.size(1),FLAGS.class_dim,FLAGS.style_dim,device)
    #     for i in range(4):
    #         #pdb.set_trace()
    #         sm,slv,cm,clv,mu_x,lv_x=training.single_pass(X, sample_crops, i, context,
    #                                                      style_mu, style_logvar, encoder, decoder, training=False)
    #         # plotting.make_crop_grid(X.clone().detach(), mu_x.clone().detach(), i, sample_crops, plot_params['epoch'],
    #         #                         plot_params['samples'], plot_params['rows'], FLAGS.logdir)
    #         context = torch.cat([cm,clv],dim=1)
    #         #print(context)
    #         #print(training.loss(FLAGS,cm,clv,sm,slv,mu_x,lv_x,X[:,reorder])[:-1])


    #     #update context but no style
    #     plot_params = {'plot':True, 'samples':10*total_per_dig, 'rows':2*total_per_dig,
    #                    'epoch':'Context_NoStyle', 'iter':'Plot', 'grp':None}   #larger means later in process
    #     context,style_mu,style_logvar = cNs_init(X.size(0),X.size(1),FLAGS.class_dim,FLAGS.style_dim,device)
    #     for i in range(4):
    #         _,_,cm,clv,mu_x,lv_x=training.single_pass(X, sample_crops, i, context,
    #                                                      style_mu, style_logvar, encoder, decoder, training=False,style=False)
    #         # plotting.make_crop_grid(X.clone().detach(), mu_x.clone().detach(), i, sample_crops, plot_params['epoch'],
    #         #                         plot_params['samples'], plot_params['rows'], FLAGS.logdir)
    #         context = torch.cat([cm,clv],dim=1)
    #         #print(context)
    #         #print(training.loss(FLAGS,cm,clv,sm,slv,mu_x,lv_x,X[:,reorder])[:-1])


    #     #Without updated context
    #     plot_params = {'plot':True, 'samples':10*total_per_dig, 'rows':2*total_per_dig,
    #                    'epoch':'NoContext_Style', 'iter':'Plot', 'grp':None}   #larger means later in process
    #     context,style_mu,style_logvar = cNs_init(X.size(0),X.size(1),FLAGS.class_dim,FLAGS.style_dim,device)
    #     for i in range(4):
    #         sm,slv,cm,clv,mu_x,lv_x=training.single_pass(X, sample_crops, i, context,
    #                                                      style_mu, style_logvar, encoder, decoder, training=False)
    #         style_mu = torch.zeros(X.size(0),X.size(1),FLAGS.style_dim)
    #         style_logvar = torch.ones(X.size(0),X.size(1),FLAGS.style_dim)
    #         # plotting.make_crop_grid(X.clone().detach(), mu_x.clone().detach(), i, sample_crops, plot_params['epoch'],
    #         #                         plot_params['samples'], plot_params['rows'], FLAGS.logdir)
    #         #print(training.loss(FLAGS,cm,clv,sm,slv,mu_x,lv_x,X[:,reorder])[:-1])

    #single sample
    #reorder = np.random.permutation(4)
    #reorder = np.array([0,1,2,3])



    #fig,axes = plt.subplots(1,1)
    actionlist = ds_test.df['action'].unique()
    #plt.gca().set_aspect('equal', adjustable='box')
    #pdb.set_trace()
    for x,smp_ix in enumerate(range(0,len(X))):
        context_ar=[]
        style_ar=[]
        with torch.no_grad():
            smpX = X[smp_ix].unsqueeze(0)
            smpCrp = sample_crops[smp_ix].unsqueeze(0)
            #With updated context
            # plot_params = {'plot':True, 'samples':10*total_per_dig, 'rows':2*total_per_dig,
            #                 'epoch':f'{smp_ix}_Context', 'iter':'Plot', 'grp':None}   #larger means later in process


            context,style_mu,style_logvar = cNs_init(smpX.size(0),smpX.size(1),FLAGS.class_dim,FLAGS.style_dim,device)
            for i in range(4):
                sm,slv,cm,clv,mu_x,lv_x=training.single_pass(smpX, smpCrp, i, context, style_mu, style_logvar, encoder, decoder, training=False)
                #plotting.make_crop_grid(smpX.clone().detach(), mu_x.clone().detach(), i, smpCrp, plot_params['epoch'],
                #                        plot_params['samples'], plot_params['rows'], FLAGS.logdir)
                context = torch.cat([cm,clv],dim=1)

                #print(training.loss(FLAGS,cm,clv,sm,slv,mu_x,lv_x,smpX)[:-1])

                #if i==1:
                    #content_test(FLAGS,context[:],smpX[0,-1-(i+1)].unsqueeze(0),smp_ix,axes)
                #print(context)

                context_ar.append(context)
                style_ar.append(torch.cat([sm,slv],dim=1))
            cnt_ten = torch.cat(context_ar).detach().cpu()
            stl_ten = torch.cat(style_ar).detach().cpu()
            print(actionlist[torch.where(smpCrp==1)[2].cpu()])
            plotting.paolas_prop(cnt_ten[:,:int(cnt_ten.shape[1]/2)].T,torch.exp(0.5*cnt_ten[:,int(cnt_ten.shape[1]/2):].T),
                                 FLAGS.logdir,'content',f'obj_{sample_targs[smp_ix]}',smp_ix,
                                 act_list = actionlist[torch.where(smpCrp==1)[2].cpu()][::-1])
            #plotting.paolas_prop(stl_ten[:,:int(stl_ten.shape[1]/2)].T,torch.exp(stl_ten[:,int(stl_ten.shape[1]/2):].T)**0.5,FLAGS.logdir,'style','',smp_ix)

#%%

path = os.path.join(os.environ['HOME'],'cluster/robot_haptic_perception/run1')
listfiles = os.listdir(path)
for f_id in listfiles:
    if (f_id[:3]=='May') or (f_id[:3]=='Jun'):
        # if '2_00-4' in f_id:
        #     continue
        # if 'Jun02_00-07-36_g055' not in f_id:
        #     continue
        #if 'Jun02_15-49-57_g094' not in f_id:
        #    continue
        #if 'Jun07_23-26-32_g09334' not in f_id: continue
        #_,_,_,_=setup_models(os.path.join(path,f_id))
        plot_everything(os.path.join(path,f_id),ds_test)
