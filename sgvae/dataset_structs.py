#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:09:21 2022

@author: richardson
"""
import os.path
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from sgvae.property_maps import property_df,get_num_classes

import pdb


class tactile_explorations(Dataset):
    def __init__(self, config, train=True, dataset='all',action_select=None):
        ds = 'train_df' if train else 'test_df'
        if dataset=='orig':
            ds = ds+'_objs'
        if dataset=='new':
            ds = ds+'_newobjs'
        #path2data = f'data/{ds}'
        df = pd.read_pickle(os.path.join(config.data_path,ds))

        if action_select is not None:
            df = df[df['action']==action_select]
        if 'index' in df.columns:
            df = df.drop('index',axis=1)
        self.df = df.reset_index()
        self.df.rename(columns={'index':'sample_ix'},inplace=True)
        self.context_ixs = None

        #action 1hot encodings
        self.act_list = self.df['action'].unique()
        self.one_hot_action = torch.zeros(len(self.df),len(self.act_list))
        for i,act in enumerate(self.act_list):
            self.one_hot_action[(self.df['action']==act).values,i]=1.0

        self.repeats = config.action_repetitions
        # if transform=='compute':
        #     self.transform = compute_transform(self.df)
        #     self.data = self.apply_transform()
        # elif transform is not None:
        #     self.transform = transform
        #     self.data = self.apply_transform()
        # self.context_ixs = None


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index=[index]
        if self.context_ixs is not None:
            ixs = self.context_ixs[index].flatten()
            ixs = ixs[torch.randperm(len(ixs))]
            index.extend(ixs.tolist())

        return self.data[index], self.one_hot_action[index], int(self.df.iloc[index[0]]['object'])

    def apply_transform(self):
        df = self.df
        df = df.drop(['kms40/time','reflex/time','ur5/time','sample_ix','run','object','action','pass_num'],axis=1)
        count,cnt_df = 0,df.iloc[0]
        self.col_order = df.columns
        for col in self.col_order:
            if len(cnt_df[col].shape)==1: count+=1
            else: count+=cnt_df[col].shape[0]
        data=torch.zeros(df.shape[0],count,len(cnt_df['kms40/fx_dc']))

        cnt=0
        for col in self.col_order:
            if col in self.transform.keys():
                mn,rng=self.transform[col]
            else:
                mn,rng=0.0,1.0
            col_dat = torch.tensor(np.stack(df[col].values))
            # if '_ac' in col:
            #     col_dat = torch.log10(torch.exp(col_dat)+1e-7)
            #### actual transform is here ####
            vals = (col_dat - mn)/rng
            #vals = 2.0*(col_dat - mn)/rng - 1.0
            if len(vals.shape)==2:
                data[:,cnt,:]=vals
                cnt+=1
            elif len(vals.shape)==3:
                data[:,cnt:cnt+vals.shape[1],:]=vals
                cnt+=vals.shape[1]
        return data

    def get_transform(self):
        return self.transform

    def set_transform(self,transform=None):
        if transform is None:
            self.transform = compute_transform(self.df)
        else:
            self.transform = transform
        self.data = self.apply_transform()

    def random_context_sampler(self):
        df = self.df
        self.context_ixs=torch.zeros(df.shape[0],
                                     EXPLORATORY_PROCEDURE_NUM*self.repeats-1,
                                     dtype=torch.long)
        for obj in df['object'].unique():
            tdf = df[(df['object'] == obj)]

            act_dict={}
            for act in df['action'].unique():
                act_dict[act] = tdf.index[tdf['action'] == act].tolist()

            for act_1 in act_dict.keys():
                act1ix = act_dict[act_1]
                cnt=0
                for act_x in act_dict.keys():
                    samp_num = self.repeats
                    if act_x==act_1: samp_num -= 1
                    for _ in range(samp_num):
                        sample_ixs = random.SystemRandom().choices(act_dict[act_x],
                                                                k=len(act1ix))
                        sample_ixs = torch.tensor(sample_ixs)
                        self.context_ixs[act1ix,cnt]=sample_ixs
                        cnt+=1


    def remove_context(self):
        self.context_ixs = None

    def set_indices(self,indices):
        self.one_hot_action = self.one_hot_action[indices]
        self.df = self.df.iloc[indices]
        self.df = self.df.reset_index(drop=True)
        self.remove_context()

def split_indices(dataset,split_ratio,split_type):
    df = dataset.df
    train_indices,val_indices = [],[]
    if split_type=='all':
        for act in df['action'].unique():
            for obj in df['object'].unique():
                td_ix = df[(df['action']==act) & (df['object']==obj)].index.tolist()
                random.shuffle(td_ix)
                split = int(len(td_ix)/split_ratio)
                train_indices.extend(td_ix[split:])
                val_indices.extend(td_ix[:split])
    elif split_type in ['orig','new']:
        obj_list = df['object'].unique()
        random.shuffle(obj_list)
        train_objs = obj_list[int(len(obj_list)/split_ratio):]
        val_objs = obj_list[:int(len(obj_list)/split_ratio)]
        train_indices = df[df.apply(lambda x: int(x['object'] in train_objs), axis=1)==True].index.tolist()
        val_indices = df[df.apply(lambda x: int(x['object'] in val_objs), axis=1)==True].index.tolist()

    return train_indices,val_indices

def compute_transform(df):
    data_types =['f._dc','f._ac','t._dc','t._ac',
                'ur5/effort','ur5/position','ur5/velocity',
                'tactile','motor_load','motor_velocity',
                'proximal','motor_angle']

                #not sure angles ('proximal','motor_angle') need normalization
                #don't normalize imu quaternions
    transform_dict = {}
    for dtp in data_types:
        mu,std,mn,rng = data_char(df,dtp)
        for col in df.filter(regex=dtp).columns:
            transform_dict[col] = torch.tensor([mn,rng])

    return transform_dict


def data_char(dataframe,data_type):
    all_dt = np.concatenate(dataframe.filter(regex=data_type).values.flatten()).flatten()
    # if '_ac' in data_type:
    #     lim = 1e-7
    #     all_dt = np.exp(all_dt)
    #     all_dt = np.log10(all_dt+lim)
    mn = all_dt.min()
    rng = np.ptp(all_dt)
    mu = all_dt.mean()
    std = all_dt.std()

    return mu,std,mn,rng

# class full_classification(tactile_explorations):
#     def __init__(self, config, train=True, transform=None, dataset='all', action_select=None):
#         super().__init__(config, train, dataset, action_select)
#         self.transform = compute_transform(self.df)
#         self.data = self.apply_transform()
#         self.property_values = property_df()

#     def __getitem__(self, index):
#         ball_id = int(self.df.iloc[index]['object'])

#         label = self._label
#         if 'contents' in label:
#            label = f'{label}_label'
#         #elif self.label in ['sq_size','pr_size','mass','stiffness']:
#         #    label = f'{self.label}_cluster_label'
#         #else: raise ValueError()
#         prop = self.property_values.loc[ball_id][label]

#         return self.data[index], self.one_hot_action[index], ball_id, prop

#     @property
#     def label(self):
#         return self._label
#     @label.setter
#     def label(self,label):
#         if label not in ['sq_size','pr_size','mass','stiffness','contents_fine','contents_rough','ball_id','contents_binary']:
#             raise ValueError(f"Must be in {['sq_size','pr_size','mass','stiffness','contents_fine','contents_rough','ball_id','contents_binary']}.")
#         self._label = label

#     @property
#     def class_cnt(self):
#         return get_num_classes(self.label,self.property_values)


class latent_representations(Dataset):
    """Dataset for storing latent representations of samples in a 
    trained model. The dataset is initialized empty and can be added to
    using the available methods. 

    :param int config.action_repetitions: The number of time each action 
    is repeated in a training sequence.
    :param int config.style_dim: The dimensionality of the style latent space.
    """


    def __init__(self, config):
        self.sequence_len = config.action_repetitions*EXPLORATORY_PROCEDURE_NUM
        self.style = (config.style_dim > 0)
        self.property_values = property_df()
        columns = [f'content {i}' for i in range(self.sequence_len)]
        if self.style:
            columns.extend(f'style {i}' for i in range(self.sequence_len))
        columns.extend(f'action {i}' for i in range(self.sequence_len))
        columns.extend(['object'])
        columns.extend(col for col in self.property_values.columns)
        self.data = pd.DataFrame(columns=columns).astype(object)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        feats = torch.squeeze(row[f'{self._latent_type} {self._iter}']).float()
        ball_ids = row['object']

        labels = torch.tensor(row[self._label]).float()
        if 'contents' in self._label:
            labels = labels.long()

        actions = row[f'action {self._iter}']
        # np.array([np.where(a==self.action_list)[0].item() for a in row[f'action {iter}']])

        return feats,actions,ball_ids,labels

    def start_row(self,obj):
        self.row = pd.DataFrame(columns=self.data.columns)
        self.row['object'] = obj
        self.row[self.property_values.columns] = self.property_values.loc[obj].values

    def add_to_row(self,cont,styl,act,ix):
        self.row[f'content {ix}'] = torch.tensor_split(cont,cont.shape[0])
        if self.style:
            self.row[f'style {ix}'] = torch.tensor_split(styl,styl.shape[0])
        self.row[f'action {ix}'] = act

    def append_row(self):
        self.data = self.data.append(self.row,ignore_index=True)

    @property
    def label(self):
        return self._label
    @label.setter
    def label(self, value):
        if value not in self.property_values.columns[1:]:
            raise ValueError(f"Must be in {self.property_values.columns[1:]}.")
        self._label = value

    @property
    def latent_type(self):
        return self._latent
    @latent_type.setter
    def latent_type(self, value):
        if value not in ['content','style']:
            raise ValueError('Must be "content" or "style".')
        self._latent_type = value

    @property
    def iteration(self):
        return self._iter
    @iteration.setter
    def iteration(self,iter):
        if iter == 'last':
            self._iter = self.sequence_len - 1
        elif (iter < 0) or (iter > self.sequence_len - 1):
            raise ValueError
        else:
            self._iter = iter

    @property
    def class_cnt(self):
        return get_num_classes(self.label,property_df())

    @property
    def data_columns(self):
        return self.data.columns

    # def apply_transform(self):

    #     return {}

    # def get_transform(self):
    #     return self.transform

    # def set_transform(self,transform=None):
    #     if transform is None:
    #         self.transform = compute_transform(self.df)
    #     else:
    #         self.transform = transform
    #     self.data = self.apply_transform()
