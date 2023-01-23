#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:31:01 2022

@author: richardson
"""

#load dataset
#split training into trianing and val
#train network on property regression (use encoder than mlp or something)
import os
import os.path
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_structs import full_classification,split_indices
from tensorboardX import SummaryWriter

import pdb

class class_net(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(class_net, self).__init__()

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
        mlp_sz = 100
        self.lin1 = nn.Sequential(nn.Linear(in_features=kwargs['hidden_dims'][-1]*cos, out_features=mlp_sz, bias=True),
                                  nn.LeakyReLU())

        # style
        self.out = nn.Linear(in_features=mlp_sz, out_features=num_classes, bias=True)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1)
        x = self.out(self.lin1(x))
        if self.num_classes==1:
            x=x.flatten()
        return x

def load_datasets(FLAGS):
    ds_train = full_classification(train=True,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    ds_val = full_classification(train=True,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    train_indices, val_indices = split_indices(ds_train,4,FLAGS.dataset)

    ds_train.set_indices(train_indices)
    ds_train.set_transform()

    ds_val.set_indices(val_indices)
    ds_val.set_transform(ds_train.get_transform())
    ds_test = full_classification(train=False,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    ds_test.set_transform(ds_train.get_transform())

    return ds_train,ds_val,ds_test

def process_batch(FLAGS,data,labels,network,optimizer,loss_func,training=True):
    optimizer.zero_grad()
    #pdb.set_trace()
    data = data.to(FLAGS.device)
    if network.num_classes==1:
        labels = labels.float().to(FLAGS.device)
    else:
        labels = labels.long().to(FLAGS.device)


    classifier_pred = network(data)

    classification_error = loss_func(classifier_pred,labels)

    if training:
        classification_error.backward()
        optimizer.step()

    #_, classifier_pred = torch.max(classifier_pred, 1)
    #classifier_accuracy = (classifier_pred.data == labels).sum().item() / FLAGS.batch_size
    return classification_error, classifier_pred
    #return classification_error, classifier_accuracy, classifier_pred

def training_loop(FLAGS,train_loader,val_loader,network,optimizer,loss_func,writer):
    best_train_loss = 100
    #best_train_acc = 0
    best_val_loss = 100
    for ep in range(FLAGS.end_epoch):
        train_accuracy,train_loss = 0,0
        for data,actions,objects,labels in train_loader:
            #cle,cla,_ = process_batch(FLAGS,data,labels,network,optimizer,loss_func)
            cle,_ = process_batch(FLAGS,data,labels,network,optimizer,loss_func)


            train_loss += cle
            #train_accuracy += cla
        writer.add_scalar('Train/Loss',train_loss/len(train_loader),ep)
        #writer.add_scalar('Train/Accuracy',train_accuracy/len(train_loader),ep)

        #validation
        #torch.save(classifier.state_dict(), os.path.join(save_dir,f'e{ep}'))
        val_loss,_,_,_,_ = eval_classifier(FLAGS,val_loader,network,optimizer,loss_func)
        #print(f'---- Val Acc = {val_acc}')
        writer.add_scalar('Val Loss',val_loss,ep)
        if val_loss < best_val_loss:
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_model = copy.deepcopy(network)
    return best_train_loss / len(train_loader), \
            best_val_loss / len(val_loader),best_model

def eval_classifier(FLAGS,loader,network,optimizer,loss_func):
    with torch.no_grad():
        loss = 0
        all_predictions=[]
        all_labels=[]
        all_actions=[]
        all_objects=[]
        for data,actions,objects,labels in loader:
            cl,pred = process_batch(FLAGS,data,labels,network,
                                      optimizer,loss_func,False)
            loss += cl
            all_predictions.append(pred)
            all_labels.append(labels)
            all_actions.append(actions)
            all_objects.append(objects)
        all_actions = torch.cat(all_actions)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_objects = torch.cat(all_objects)

    return loss / len(loader), all_predictions.cpu(), all_labels.cpu(), all_actions.cpu(), all_objects

def run_all(FLAGS):
    train,val,test = load_datasets(FLAGS)

    #loaders
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(train,batch_size=FLAGS.batch_size,shuffle=True,**kwargs)
    val_loader = DataLoader(val,batch_size=FLAGS.batch_size,shuffle=True,**kwargs)
    test_loader = DataLoader(test,batch_size=FLAGS.batch_size,**kwargs)

    if type(FLAGS.test_properties) is not list:
        FLAGS.test_properties = [FLAGS.test_properties]
    for prop in FLAGS.test_properties:
        log_dir = 'baseline_classification_results'
        if FLAGS.save_folder is not None:
            if FLAGS.action_select is not None:
                log_dir = os.path.join(log_dir,f'{int(FLAGS.save_folder)}_{FLAGS.action_select}')
            else:
                log_dir = os.path.join(log_dir,f'{int(FLAGS.save_folder)}')
        log_dir = os.path.join(log_dir,f'{prop}_{int(FLAGS.save_dir_id)}')
        writer=SummaryWriter(log_dir=log_dir)
        train.set_label(prop)
        val.set_label(prop)
        test.set_label(prop)
        if 'contents' in prop:
            loss_func = nn.CrossEntropyLoss()
            class_cnt = train.get_class_cnt()
        else:
            loss_func = nn.MSELoss()
            class_cnt = 1

        net = class_net(num_classes = class_cnt,
                        in_channels = FLAGS.in_channels,
                        hidden_dims = FLAGS.hidden_dims,
                        kernels = FLAGS.kernels,
                        strides = FLAGS.strides,
                        paddings = FLAGS.paddings).to(FLAGS.device)
        optimizer = optim.Adam(
            list(net.parameters()),
            lr=FLAGS.initial_learning_rate,
            betas=(FLAGS.beta_1, FLAGS.beta_2),
            weight_decay=1)
        tr_loss,val_loss,best_model = training_loop(FLAGS,train_loader,val_loader,net,optimizer,loss_func,writer)
        ts_loss,preds,lbls,acts,objs = eval_classifier(FLAGS,test_loader,best_model,optimizer,loss_func)
        torch.save(best_model.state_dict(),os.path.join(log_dir,'model'))
        save_dict = {'train_loss':tr_loss,'val_loss':val_loss,
                     'test_loss':ts_loss,'test_preds':preds,
                     'test_lbls':lbls,'test_actions':acts,
                     'test_objs':objs}
        torch.save(save_dict, os.path.join(log_dir,'results_dict'))





parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")
parser.add_argument('--end_epoch', type=int, default=200, help="flag to indicate the final epoch of training")
parser.add_argument('--in_channels', type=int, default=195)
parser.add_argument('--hidden_dims', type=int, nargs='*', default=[128,64,64,32])
parser.add_argument('--kernels', type=int, nargs='*', default=[6,6,5,4])
parser.add_argument('--strides', type=int, nargs='*', default=[4,4,2,2])
parser.add_argument('--paddings', type=int, nargs='*', default=[1,1,1,0])
parser.add_argument('--initial_learning_rate', type=float, default=0.00001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
# 'size','mass','stiffness','contents_fine','contents_rough','ball_id','contents_binary'
parser.add_argument('--dataset', type=str, default='new', choices=['all','objects','new'])
parser.add_argument('--save_dir_id', type=str, default=None)
parser.add_argument('--test_properties', type=str, nargs='*', default=['pr_size','sq_size','stiffness','mass','contents_binary'])
parser.add_argument('--action_select', type=str, default=None, choices=['squeeze', 'press', 'shake', 'slide'])
parser.add_argument('--save_folder', type=str, default=None)

FLAGS = parser.parse_args()

if __name__ == '__main__':
    FLAGS.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_all(FLAGS)
