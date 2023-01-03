#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:15:17 2022

@author: richardson
"""

import matplotlib.pyplot as plt
import ball_datasets as bd
import multiprocessing as mp
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torch
import os
import os.path
import yaml
from PIL import Image
import property_maps
import pdb
from classification_accuracy import create_datasets,setup_dataset

def plot_multiple(X,mu_x,acts,objs,cs,plot_params,act_list,path):
    sav_path = os.path.join(path,f'EP-{plot_params["epoch"]}')
    pool = mp.Pool(4)
    if not os.path.isdir(sav_path):
        os.mkdir(sav_path)
    for ex in range(X.size(0)):
        obj = objs[ex]
        pool.starmap(plot_XXp_wrapper,[(i,ex,X,mu_x,act_list,acts,
                                       obj,cs,plot_params['cols'],
                                       sav_path) for i in range(X.size(1))])
    pool.close()
        # for i in range(X.size(1)):
        #     x1 = X[ex,-1-i]
        #     x2 = mu_x[ex,-1-i]
        #     act = act_list[torch.where(acts[ex,-1-i]==1.)[0]]
        #     title = f'{ex}ix-Num-{i}_ExSeen-{cs}_A-{act}_O-{obj}'
        #     with plt.ioff():
        #         plot_XXp(x1,x2, plot_params['cols'],title, sav_path)

def plot_XXp_wrapper(i,ex,X,mu_x,act_list,acts,obj,cs,cols,sav_path):
    x1 = X[ex,-1-i]
    x2 = mu_x[ex,-1-i]
    act = act_list[torch.where(acts[ex,-1-i]==1.)[0]]
    title = f'{ex}ix-Num-{i}_ExSeen-{cs}_A-{act}_O-{obj}'
    with plt.ioff():
        plot_XXp(x1,x2,cols,title, sav_path,act)

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
            #ax[i,j].set_ylim([-1.1,1.1])

            i = (i+1)%6
            cnt+=1
            if ('_dc' not in col) and (i==0): j+=1
    for axs in ax.flatten():
        axs.set_xticks([])
        axs.set_yticks([])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2,hspace=0.1)
    fig.suptitle(title)

    fig.savefig(f'{sav_path}/{title}.png',dpi=350,bbox_inches='tight',pad_inches=0)
    plt.close(fig)

def paolas_prop(mu_mat,sigma_mat,tmp_folder,epoch,ix,sample_num,act_list=None):
    #mu_mat = np.random.randint(0,5, size=(10,4))
    #sigma_mat = np.random.randint(1,5, size=(10,4))
    plt.ioff()
    for i in range(mu_mat.shape[0]):
        fig_int, ax_int = plt.subplots(1,mu_mat.shape[1],sharey='all')
        for j in range(mu_mat.shape[1]):
            mu = mu_mat[i,j]
            sigma = sigma_mat[i,j]
            gaussian = stats.norm(mu, sigma)
            ys = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            ax_int[j].plot(gaussian.pdf(ys), ys, color='deepskyblue', label='gaussian normal')
            ax_int[j].plot(0, mu, marker='o', color='crimson', label='mean')
            ax_int[j].set_ylim(-4, 4)
            ax_int[j].spines.right.set_visible(False)
            ax_int[j].spines.top.set_visible(False)
            ax_int[j].set_xticks([])
            ax_int[j].set_xlabel(act_list[j])


            if j>0:
                #ax_int[j].spines.left.set_visible(False)
                ax_int[j].axes.get_yaxis().set_visible(False)

            fig_int.subplots_adjust(wspace=0.1)
        plt.savefig(str(i) + '.png',dpi=350)
        plt.close()

    files = sorted([f for f in os.listdir(os.getcwd()) if
                         f.endswith(".png") and f!="result.png"])

    images = [Image.open(x) for x in files]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save(f'{tmp_folder}/distributions_{sample_num}_{epoch}_{ix}.png')

    [os.remove(f) for f in files]
    plt.close()

    plt.ion()
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

def plot_content_points_by_property(contents,actions,labels,prop,cnt,discrete=False,last=False):
    action_markers = {'press':'v','squeeze':'o','shake':'+','slide':'4'}
    if discrete==False:
        cm = 'viridis'
    else:
        sltc = plt.cm.tab20b([3,2,1,0])
        ppcc = plt.cm.tab20b([11,10,9,8])
        stnc = plt.cm.tab20c([19,18,17,16])
        othc = plt.cm.tab20c([1,9])  # none, loose
        cm = np.vstack((othc,sltc,ppcc,stnc))
        cm = np.array([[1,133,113,256],
                       [255, 255, 0, 256],
                       [251,180,185,256],
                       [247,104,161,256],
                       [197,27,138,256],
                       [122,1,119,256],
                       [253.,190,133,256],
                       [253,141,60,256],
                       [230,85,13,256],
                       [166,54,3,256],
                       [189,215,231,256],
                       [107,174,214,256],
                       [49,130,189,256],
                       [8,81,156,256]] )/256.
        cm = LinearSegmentedColormap.from_list('my_colormap', cm,N=len(cm))
        named_labels = labels.copy()
        
        cm = LinearSegmentedColormap.from_list('my_colormap', np.array([[1,133,113,256],[8,81,156,256]])/256.,N=2)
        # for i,l in enumerate(labels):
        #     if l=='none': 
        #         labels[i]=0
        #     # if l=='loose': labels[i]=1
        #     # if 'salt' in l: labels[i]=1+int(l[-1])
        #     # if 'popcorn' in l: labels[i]=5+int(l[-1])
        #     # if 'rocks' in l: labels[i]=9+int(l[-1])
        #     else:
        #         labels[i]=1
    if prop=='mass':
        labels = np.log2(labels.astype(float)+1)


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.title(prop)

    for i in range(contents.shape[1]):
        if last:
            order = np.argsort(labels)
            i=contents.shape[1]-1
        img = ax.scatter(contents[order,i,0],contents[order,i,1],c=labels[order],
                         cmap=cm,s=150,zorder=10,linewidth=1, edgecolor='k')
        ax.tick_params(left=False, bottom=False,
                       labelleft=False,
                       labelbottom=False)
        for _, sp in ax.spines.items():
            sp.set_linewidth(5)
        if last: break

        if i>0:
            x = contents[:,i-1,0]
            y = contents[:,i-1,1]
            dx = contents[:,i,0]-contents[:,i-1,0]
            dy = contents[:,i,1]-contents[:,i-1,1]
            ax.quiver(x,y,dx,dy,angles='xy', scale_units='xy', scale=1.2, width=.01,zorder=1,
                      headlength=3,headwidth=3,headaxislength=3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(img, cax=cax)
    cbar.set_ticks([])

    plt.savefig(f'corl_figures/latent_points_2D/{cnt}points_{"last" if last else "all"}_{prop}.png',dpi=350)
    plt.savefig(f'corl_figures/latent_points_2D/{cnt}points_{"last" if last else "all"}_{prop}.svg',dpi=350)


def plot_classification_perf(prop,beta=0.033,
                             modbase = os.path.join(os.environ['HOME'],'cluster/robot_haptic_perception'),
                             base = os.path.join(os.environ['HOME'],'cluster/robot_haptic_perception'),
                             plot_confusion_matrix=False):
    results_folder = os.path.join(base,'classification_results')
    model_base = os.path.join(modbase,'runs')
    all_folds = os.listdir(results_folder)
    results = []
    for fld in all_folds:
        config = os.path.join(model_base,fld[:-2],'config.yaml')
        row = {}
        with open(config, 'r') as file:
            config = yaml.load(file,Loader=yaml.UnsafeLoader)
        row['beta'] = config.beta
        row['vae_mod_fld'] = fld[:-2]
        class_res = torch.load(os.path.join(results_folder,fld,prop,'class_dict'))
        style_res = torch.load(os.path.join(results_folder,fld,prop,'style_dict'))
        for key in class_res:
            if key == 'train_acc':
                row[f'class_{key}'] = class_res[key]/814
                row[f'style_{key}'] = style_res[key]/814
            else:
                row[f'class_{key}'] = class_res[key]
                row[f'style_{key}'] = style_res[key]
        results.append(row)
        props = pd.read_csv(os.path.join(results_folder,fld,'properties.csv'))
    results = pd.DataFrame(results)
    results = results[results['beta']==beta]

    for tp in ['class']: #,'style']:
        if plot_confusion_matrix==False:
            tra = np.stack(results[f'{tp}_test_acc'])
            xE = np.stack(results[f'{tp}_test_entropy'])
            xA = np.concatenate([results[[f'{tp}_train_acc',f'{tp}_val_acc']].values,tra],axis=1)

            xMP = np.stack(results[f'{tp}_test_probs'].values).max(axis=3).mean(axis=1)
            x2MP = xMP - np.sort(np.stack(results[f'{tp}_test_probs'].values))[:,:,:,-2].mean(axis=1)

            fig,axs = plt.subplots(1,3,figsize=(11,8))
            fig,axs = plt.subplots(1,2,figsize=(11,8))

            fig.suptitle(f'{str.capitalize(tp)} {str.capitalize(prop)}')
            fig.tight_layout()
            plt.rc('font', size=25)
            lw = dict(linewidth=3)
            fw = dict(markersize=10,markeredgewidth=3)
            axs[0].boxplot(xA,boxprops=lw,medianprops=lw,whiskerprops=lw,capprops=lw,flierprops=fw,widths=0.3,
                            positions = [1,1.5,2,2.5,3,3.5])
            axs[0].set_ylim(0,1)
            axs[0].set_xticks([1,1.5,2,2.5,3,3.5], ['Train', 'Val', 'Test_1', 'Test_2', 'Test_3', 'Test_4'],rotation=60)
            axs[0].set_title('Accuracy')
            axs[1].boxplot(xE,boxprops=lw,medianprops=lw,whiskerprops=lw,capprops=lw,flierprops=fw,widths=0.3,
                            positions=[1,1.5,2,2.5])
            axs[1].set_xticks([1,1.5,2,2.5], ['Test_1', 'Test_2', 'Test_3', 'Test_4'],rotation=60)
            axs[1].set_title('Entropy')
            # axs[2].boxplot(x2MP,boxprops=lw,medianprops=lw,whiskerprops=lw,capprops=lw,flierprops=fw,widths=0.3,
            #                positions=[1,1.5,2,2.5])
            # axs[2].set_xticks([1,1.5,2,2.5], ['Test_1', 'Test_2', 'Test_3', 'Test_4'],rotation=60)
            # axs[2].set_title('Max Prob')
            ### appearance
            for ix in range(2):
                for axis in ['top','bottom','left','right']:
                    axs[ix].spines[axis].set_linewidth(3)
                axs[ix].xaxis.set_tick_params(width=3)
                axs[ix].yaxis.set_tick_params(width=3)
            plt.savefig(f'figures/classification_results/{beta}_{tp}_{prop}.svg',dpi=350)
            plt.close()
        else:
        #confusion matrix
            test_probs = torch.tensor(np.concatenate(results[f'{tp}_test_probs']))
            lbls = np.concatenate(results[f'{tp}_labels'])
            preds = test_probs[:,3].max(dim=1).indices
            confm = confusion_matrix(lbls,preds, normalize='true')
            names = props['ball_name'].values

            for i in range(len(names)):
                if '_popcrn' in names[i]:
                    names[i] = names[i][:-6]+'popcorn'
                elif 'popcrn_' in names[i]:
                    names[i] = 'popcorn'+names[i][6:]
            df_cm = pd.DataFrame(confm, index=props['ball_name'].values, columns=props['ball_name'].values)
            # sort_order = ['baseball_1', 'baseball_2', 'beige_1', 'blue_green_squishy',
            #        'blue_resistance', 'cork_miner', 'empty', 'foam_1', 'foam_2', 'foam_3',
            #        'foam_4', 'globe_squishy', 'golfball_1', 'golfball_2', 'hard_plastic_l',
            #        'inflate_globe', 'lacrosse', 'orange_foam_stf', 'pingpong_1',
            #        'popcorn_s', 'popcorn_m', 'popcorn_l', 'popcorn_xl', 'purple_foam_stf',
            #        'purple_resistance', 'purple_spiky', 'purple_squash_1',
            #        'purple_squash_2', 'purple_squash_3', 'red_squishy', 'salt_s', 'salt_m',
            #        'salt_l', 'salt_xl', 'smooth_foam_s', 'smooth_foam_m', 'smooth_foam_l',
            #        'smooth_foam_xl', 'stone_s', 'stone_m', 'stone_l', 'stone_xl',
            #        'tennis_1', 'tennis_2', 'tennis_3', 'yellow_foam_stf', 'yellow_hockey',
            #        'yellow_soft_xs','yellow_soft_s', 'yellow_soft_m', 'yellow_soft_l',
            #        'yellow_soft_popcorn' ]
            sort_order = ['empty','baseball_1', 'baseball_2', 'golfball_1', 'golfball_2',
                          'foam_1', 'foam_2', 'foam_3','foam_4',
                          'purple_squash_1','purple_squash_2', 'purple_squash_3',
                          'tennis_1', 'tennis_2', 'tennis_3',
                          'popcorn_s', 'popcorn_m', 'popcorn_l', 'popcorn_xl',
                          'salt_s', 'salt_m','salt_l', 'salt_xl',
                          'stone_s', 'stone_m', 'stone_l', 'stone_xl',
                          'smooth_foam_s', 'smooth_foam_m', 'smooth_foam_l',
                          'smooth_foam_xl',
                          'yellow_soft_xs','yellow_soft_s', 'yellow_soft_m', 'yellow_soft_l',
                          'beige_1', 'blue_green_squishy',
                           'blue_resistance','purple_resistance', 'cork_miner',
                           'globe_squishy',  'hard_plastic_l',
                           'inflate_globe', 'lacrosse',  'pingpong_1',
                            'orange_foam_stf','purple_foam_stf','yellow_foam_stf',
                            'purple_spiky',  'red_squishy',
                            'yellow_hockey',
                           'yellow_soft_popcorn' ]
            df_cm=df_cm.reindex(index=sort_order,columns=sort_order)

            plt.rc('font', size=10)
            fig,ax = plt.subplots(figsize=(11,11))
            sns.heatmap(df_cm, cmap='Oranges', annot=False,
                        xticklabels=True, yticklabels=True,
                        square = True,cbar_kws={"shrink": 0.75}, ax=ax)

            h = ax.get_yticks()
            w = ax.get_xticks()
            #line_points = np.array([2,7,11,12,14,19,23,26,29,30,34,38,42,45,47,51])
            line_points = np.array([1,3,5,9,12,15,19,23,27,31,35])
            for pnt in line_points:
                ax.hlines(pnt,w[0]-0.5,w[-1]+0.5, linewidth=1, color='k')
                ax.vlines(pnt,h[0]-0.5,h[-1]+0.5, linewidth=1, color='k')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.tick_params('both', length=6, width=1, which='major')
            fig.tight_layout(pad=0)
            fig.savefig('corl_figures/classification_results/0.033_class_ball_id_confmat.svg',dpi=350)
            fig.savefig('corl_figures/classification_results/0.033_class_ball_id_confmat.png',dpi=350)

def plot3dprops():
    prps = property_maps.property_df(scale=False)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1,1,2))
    plt.rcParams['font.family'] = 'serif'
    plt.rc('font',size = 30)
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    test = prps.loc[[3,5,11,12,36,54,59,64,73,74,78,82,83,95,97]]

    prps = prps.drop([3,5,11,12,36,54,59,64,73,74,78,82,83,95,97],axis=0)
    ax.scatter(prps['sq_size'].values*100, prps['mass'].values, prps['stiffness'].values/1000,s=200,
               facecolor='b',edgecolor='k',alpha=0.7)
    ax.scatter(test['sq_size'].values*100, test['mass'].values, test['stiffness'].values/1000,s=200,
               facecolor='r',edgecolor='k',alpha=0.7)

    ax.set_xlabel('Size (cm)')
    ax.set_ylabel('Mass (g)')
    ax.set_zlabel('Stiffness (N/mm)')


def plot_latent_stats():
    plt.rc('font', size=25)
    plt.rc('font',**{'family':'serif','serif':['Times']})
    plt.rc('text', usetex=True)
    dire = 'runs'
    class tmp:
        dataset='new'
        action_select=None
    FLAGS = tmp()

    ds_train,ds_val,ds_test,properties = setup_dataset(FLAGS)
    tdcl,vdcl,sdcl = [],[],[]
    tdsl,vdsl,sdsl = [],[],[]
    for file in ['Aug26_19-42-33_LABKJK12']:
        load_path = os.path.join(dire,file)
        print('generating latent representation dataset')
        td,vd,sd = create_datasets(load_path,ds_train,ds_val,ds_test,properties,4)

        tdcl.append(torch.exp(0.5*torch.tensor(np.stack(td.data['clv'].values))))
        vdcl.append(torch.exp(0.5*torch.tensor(np.stack(vd.data['clv'].values))))
        sdcl.append(torch.exp(0.5*torch.tensor(np.stack(sd.data['clv'].values))))

        tdsl.append(torch.exp(0.5*torch.tensor(np.stack(td.data['slv'].values))))
        vdsl.append(torch.exp(0.5*torch.tensor(np.stack(vd.data['slv'].values))))
        sdsl.append(torch.exp(0.5*torch.tensor(np.stack(sd.data['slv'].values))))

    tdcl = torch.cat(tdcl)
    vdcl = torch.cat(vdcl)
    sdcl = torch.cat(sdcl)
    tdsl = torch.cat(tdsl)
    vdsl = torch.cat(vdsl)
    sdsl = torch.cat(sdsl)

    def plotit(lat_type,dat_list):
        fig,axs = plt.subplots(1,3,figsize=(15,8))
        axs = axs.flatten()
        fig.tight_layout()
        lw = dict(linewidth=3)
        fw = dict(markersize=10,markeredgewidth=3)
        for i,dat in enumerate(dat_list):
            axs[i].boxplot(np.array(dat.mean(dim=2)),
                           boxprops=lw,
                           medianprops=lw,
                           whiskerprops=lw,
                           capprops=lw,flierprops=fw,widths=0.3,
                           positions = [1,1.5,2,2.5])
            axs[i].set_xticks([1,1.5,2,2.5],['1', '2', '3', '4'])
            axs[i].set_ylim([0,0.16])
            axs[i].set_yticks([0,0.04,0.08,0.12,0.16])
            for axis in ['top','bottom','left','right']:
                axs[i].spines[axis].set_linewidth(3)
            axs[i].xaxis.set_tick_params(width=3)
            axs[i].yaxis.set_tick_params(width=3)
        fig.savefig(f'corl_figures/{lat_type}_var.svg',dpi=350)
        fig.savefig(f'corl_figures/{lat_type}_var.png',dpi=350)

    plotit('Content',[tdcl,vdcl,sdcl])
    plotit('Style',[tdsl,vdsl,sdsl])


# def regression_perf():
#     res_df=[]
#     for i in range(5):
#         dict = {}
#         df = pd.read_pickle(f'baseline_results/{i}_None/all_results')
#         trdf = df[(df['set_name']=='train') or (df['set_name']=='val')]
#         tsdf = df[df['set_name']=='test']
#         dict['train_pr'] = ((trdf['pr_size_lbls'] - trdf['pr_size_pred'])**2)/len(df)
#         dict['train_sq'] = ((trdf['sq_size_lbls'] - trdf['sq_size_pred'])**2)/len(df)
#         dict['train_stf'] = ((trdf['stiffness_lbls'] - trdf['stiffness_pred'])**2)/len(df)
#         dict['train_ms'] = ((trdf['mass_lbls'] - trdf['mass_pred'])**2)/len(df)
#         dict['test_pr'] = ((tsdf['pr_size_lbls'] - tsdf['pr_size_pred'])**2)/len(df)
#         dict['test_sq'] = ((stdf['sq_size_lbls'] - tsdf['sq_size_pred'])**2)/len(df)
#         dict['test_stf'] = ((tsdf['stiffness_lbls'] - tsdf['stiffness_pred'])**2)/len(df)
#         dict['test_ms'] = ((tsdf['mass_lbls'] - tsdf['mass_pred'])**2)/len(df)
#         res_df.append(dict)
