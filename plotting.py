#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:15:17 2022

@author: richardson
"""

import matplotlib.pyplot as plt
import ball_datasets as bd
import multiprocessing as mp
import numpy as np
import torch
import os
import os.path

#data = bd.tactile_explorations()

#%%
def plot_multiple(X,mu_x,acts,objs,cs,plot_params,act_list,path):
    sav_path = os.path.join(path,f'EP-{plot_params["epoch"]}')
    #pool = mp.Pool(4)
    if not os.path.isdir(sav_path):
        os.mkdir(sav_path)
    for ex in range(X.size(0)):
        obj = objs[ex]
    #     pool.starmap(plot_XXp_wrapper,[(i,ex,X,mu_x,act_list,acts,
    #                                    obj,cs,plot_params['cols'],
    #                                    sav_path) for i in range(X.size(1))])
    # pool.close()
        for i  in range(X.size(1)):
            plot_XXp_wrapper(i,ex,X,mu_x,act_list,acts,
                             obj,cs,plot_params['cols'],
                             sav_path)

def plot_XXp_wrapper(i,ex,X,mu_x,act_list,acts,obj,cs,cols,sav_path):
    x1 = X[ex,-1-i]
    x2 = mu_x[ex,-1-i]
    act = act_list[torch.where(acts[ex,-1-i]==1.)[0]]
    title = f'{ex}ix-Num-{i}_ExSeen-{cs}_A-{act}_O-{obj}'
    with plt.ioff():
        plot_XXp(x1,x2,cols,title,sav_path,act)

def plot_XXp(X,Xp,col_order,title,sav_path,act):
    fig,ax = plt.subplots(6,14)
    i,j,cnt=0,0,0
    first=True
    lim=-1
    if (act=='slide') or (act=='shake'): lim = 75
    for col in col_order:
        if '_ac' in col:
            y = X[cnt:cnt+21,:lim]
            yp = Xp[cnt:cnt+21,:lim]
            cnt+=21
            ax[i,j+1].pcolormesh(np.arange(yp.shape[1]), np.arange(21), y)
            ax[i,j+2].pcolormesh(np.arange(yp.shape[1]), np.arange(21), yp)
        else:
            y = X[cnt,:lim]
            yp = Xp[cnt,:lim]
            if ('_dc' in col) and first:
                i,j=0,j+1
                first=False
            ax[i,j].plot(y)
            ax[i,j].plot(yp)
            #ax[i,j].set_ylim([-1.1, 1.1])

            i = (i+1)%6
            cnt+=1
            if ('_dc' not in col) and (i==0): j+=1
    for axs in ax.flatten():
        axs.set_xticks([])
        axs.set_yticks([])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1,hspace=0.1)
    fig.suptitle(title)

    fig.savefig(f'{sav_path}/{title}.png',dpi=350,bbox_inches='tight',pad_inches=0)
    plt.close(fig)
#%%


# col_order = ['ur5/position/shoulder_pan_joint', 'ur5/position/shoulder_lift_joint',
#            'ur5/position/elbow_joint', 'ur5/position/wrist_1_joint',
#            'ur5/position/wrist_2_joint', 'ur5/position/wrist_3_joint',
#            'ur5/velocity/shoulder_pan_joint', 'ur5/velocity/shoulder_lift_joint',
#            'ur5/velocity/elbow_joint', 'ur5/velocity/wrist_1_joint',
#            'ur5/velocity/wrist_2_joint', 'ur5/velocity/wrist_3_joint',
#            'ur5/effort/shoulder_pan_joint', 'ur5/effort/shoulder_lift_joint',
#            'ur5/effort/elbow_joint', 'ur5/effort/wrist_1_joint',
#            'ur5/effort/wrist_2_joint', 'ur5/effort/wrist_3_joint',
#            'reflex/finger_0/proximal', 'reflex/finger_0/tactile_0',
#            'reflex/finger_0/tactile_1', 'reflex/finger_0/tactile_2',
#            'reflex/finger_0/tactile_3', 'reflex/finger_0/tactile_4',
#            'reflex/finger_0/tactile_5', 'reflex/finger_0/tactile_6',
#            'reflex/finger_0/tactile_7', 'reflex/finger_0/tactile_8',
#            'reflex/finger_0/tactile_9', 'reflex/finger_0/tactile_10',
#            'reflex/finger_0/tactile_11', 'reflex/finger_0/tactile_12',
#            'reflex/finger_0/tactile_13', 'reflex/finger_0/imu_w',
#            'reflex/finger_0/imu_x', 'reflex/finger_0/imu_y',
#            'reflex/finger_0/imu_z', 'reflex/finger_0/motor_angle',
#            'reflex/finger_0/motor_velocity', 'reflex/finger_0/motor_load',
#            'reflex/finger_1/proximal', 'reflex/finger_1/imu_w',
#            'reflex/finger_1/imu_x', 'reflex/finger_1/imu_y',
#            'reflex/finger_1/imu_z', 'reflex/finger_1/motor_angle',
#            'reflex/finger_1/motor_velocity', 'reflex/finger_1/motor_load',
#            'reflex/finger_2/proximal', 'reflex/finger_2/imu_w',
#            'reflex/finger_2/imu_x', 'reflex/finger_2/imu_y',
#            'reflex/finger_2/imu_z', 'reflex/finger_2/motor_angle',
#            'reflex/finger_2/motor_velocity', 'reflex/finger_2/motor_load',
#            'reflex/preshape/motor_angle', 'reflex/preshape/motor_velocity',
#            'reflex/preshape/motor_load', 'reflex/palm/imu_w', 'reflex/palm/imu_x',
#            'reflex/palm/imu_y', 'reflex/palm/imu_z', 'kms40/fx_dc', 'kms40/fx_ac',
#            'kms40/fy_dc', 'kms40/fy_ac', 'kms40/fz_dc', 'kms40/fz_ac',
#            'kms40/tx_dc', 'kms40/tx_ac', 'kms40/ty_dc', 'kms40/ty_ac',
#            'kms40/tz_dc', 'kms40/tz_ac']

#%%
# def plot_trial(data,index=None,action=None,obj=None):
#     df = data.df
#     if index:
#         row = df.iloc[[index]]
#     else:
#         if action:
#             df = df[df['action']==action]
#         if obj:
#             df = df[df['object']==obj]
#         row = df.sample()

#     fontsize=8
#     fig, axs = plt.subplots(4,3)
#     axs = axs.flatten()
#     fig.suptitle(f'{row["object"].item()}_{row["action"].item()}')
#     for jnt in row.filter(regex='ur5/position').columns:
#         axs[0].plot(row['ur5/time'].item(),row[jnt].item(),label = jnt)
#         axs[0].xaxis.set_tick_params(labelbottom=False)
#         axs[0].set_xticks([])
#     for jnt in row.filter(regex='ur5/velocity').columns:
#         axs[1].plot(row['ur5/time'].item(),row[jnt].item(),label = jnt)
#         axs[1].xaxis.set_tick_params(labelbottom=False)
#         axs[1].set_xticks([])
#     for jnt in row.filter(regex='ur5/effort').columns:
#         axs[2].plot(row['ur5/time'].item(),row[jnt].item(),label = jnt)
#         axs[2].xaxis.set_tick_params(labelbottom=False)
#         axs[2].set_xticks([])
#     axs[0].legend(fontsize=fontsize)
#     for mtr in row.filter(regex='motor_angle').columns:
#         axs[3].plot(row['reflex/time'].item(),row[mtr].item(),label = mtr)
#         axs[3].xaxis.set_tick_params(labelbottom=False)
#         axs[3].set_xticks([])
#     axs[3].legend(fontsize=fontsize)
#     for fin in row.filter(regex='proximal').columns:
#         axs[4].plot(row['reflex/time'].item(),row[fin].item(),label = fin)
#         axs[4].xaxis.set_tick_params(labelbottom=False)
#         axs[4].set_xticks([])
#     axs[4].legend(fontsize=fontsize)
#     for mtr in row.filter(regex='motor_load').columns:
#         axs[5].plot(row['reflex/time'].item(),row[fin].item(),label = mtr)
#         axs[5].xaxis.set_tick_params(labelbottom=False)
#         axs[5].set_xticks([])
#     axs[5].legend(fontsize=fontsize)
#     for imu in row.filter(regex='finger_0/euler').columns:
#         axs[6].plot(row['reflex/time'].item(),row[imu].item(),label = imu)
#         axs[6].xaxis.set_tick_params(labelbottom=False)
#         axs[6].set_xticks([])
#     axs[6].legend(fontsize=fontsize)
#     for imu in row.filter(regex='finger_1/euler').columns:
#         axs[7].plot(row['reflex/time'].item(),row[imu].item(),label = imu)
#         axs[7].xaxis.set_tick_params(labelbottom=False)
#         axs[7].set_xticks([])
#     axs[7].legend(fontsize=fontsize)
#     for imu in row.filter(regex='finger_2/euler').columns:
#         axs[8].plot(row['reflex/time'].item(),row[imu].item(),label = imu)
#         axs[8].xaxis.set_tick_params(labelbottom=False)
#         axs[8].set_xticks([])
#     axs[8].legend(fontsize=fontsize)
#     for tac in row.filter(regex='tactile').columns:
#         axs[9].plot(row['reflex/time'].item(),row[tac].item(),label = tac)
#         axs[9].xaxis.set_tick_params(labelbottom=False)
#         axs[9].set_xticks([])
#     axs[9].legend(fontsize=fontsize)
#     for f in row.filter(regex='kms40/f').columns:
#         axs[10].plot(row['kms40/time'].item(),row[f].item(),label = f)
#         axs[10].xaxis.set_tick_params(labelbottom=False)
#         axs[10].set_xticks([])
#     axs[10].legend(fontsize=fontsize)
#     for t in row.filter(regex='kms40/t').columns:
#         if 'time' in t:
#             continue
#         axs[11].plot(row['kms40/time'].item(),row[t].item(),label = t)
#         axs[11].xaxis.set_tick_params(labelbottom=False)
#         axs[11].set_xticks([])
#     axs[11].legend(fontsize=fontsize)
