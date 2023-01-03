import os
import argparse
from itertools import cycle
import yaml
import copy

import numpy as np
import torch
import torch.special
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable

from torch.utils.data import DataLoader
from networks import Baseline_model
from utils import weights_init, reparameterize, setup_models, cNs_init
from ball_datasets import full_classification,split_indices,latent_representations
import training
from property_maps import property_df,get_num_classes
import pandas as pd

from tensorboardX import SummaryWriter

import pdb

# Set up dataset so don't have to rerun
def setup_dataset(FLAGS):
    ds_train = full_classification(train=True,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    ds_val = full_classification(train=True,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    ds_test = full_classification(train=False,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    train_indices, val_indices = split_indices(ds_train,4,FLAGS.dataset)
    ds_train.set_indices(train_indices)
    ds_val.set_indices(val_indices)
    ds_train.set_transform()
    ds_val.set_transform(ds_train.get_transform())
    ds_test.set_transform(ds_train.get_transform())
    props = property_df('cluster')
    return ds_train,ds_val,ds_test,props


def run_all(FLAGS,log_dir,tr_ds,vl_ds,ts_ds,prop):
    log_dir = os.path.join(log_dir,prop)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    FLAGS.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #setup datasets
    tr_ds.set_lbl(prop)
    vl_ds.set_lbl(prop)
    ts_ds.set_lbl(prop)

    if 'contents' in prop:
        loss_func = nn.CrossEntropyLoss()
        class_cnt = tr_ds.get_class_cnt()
    else:
        loss_func = nn.MSELoss()
        class_cnt = 1

    model = Baseline_model(output_dim = class_cnt,
                           in_channels = FLAGS.in_channels,
                           hidden_dims = FLAGS.hidden_dims,
                           kernels = FLAGS.kernels,
                           strides = FLAGS.strides,
                           paddings = FLAGS.paddings).to(FLAGS.device)
    optimizer = optim.Adam(
        list(model.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2),
        weight_decay=FLAGS.weight_decay
    )

    train_loader = DataLoader(tr_ds,batch_size=FLAGS.batch_size,shuffle=True)
    val_loader = DataLoader(vl_ds,batch_size=FLAGS.batch_size,shuffle=True)
    test_loader = DataLoader(ts_ds,batch_size=FLAGS.batch_size,shuffle=True)

    writer = SummaryWriter(log_dir=log_dir)

    _,_,best_model = train_model(FLAGS,train_loader,val_loader,
                                 model,optimizer,loss_func,writer)
    writer.close()
    torch.save(best_model.state_dict(),os.path.join(log_dir,'model'))

    # train_loader = DataLoader(tr_ds,batch_size=FLAGS.batch_size,shuffle=False)
    # val_loader = DataLoader(vl_ds,batch_size=FLAGS.batch_size,shuffle=False)
    # test_loader = DataLoader(ts_ds,batch_size=FLAGS.batch_size,shuffle=False)
    # load_dict = {'train':train_loader,'val':val_loader,
    #              'test':test_loader}
    #
    # for set_id,load in load_dict.items():
    #     loss,preds,lbls,acts,objs = test_model(FLAGS.device,load,best_model,
    #                                            loss_func)
    #     save_dict = {f'{set_id}_loss':loss,f'{set_id}_preds':preds,
    #                  f'{set_id}_lbls':lbls,f'{set_id}_actions':acts,
    #                  f'{set_id}_objs':objs}
    #     torch.save(save_dict, os.path.join(log_dir,'dict'))

    return best_model.eval()



def process_batch(device,data,labels,model,optimizer,loss_func,training=True):
    if training:
        optimizer.zero_grad()
    data = data.to(device)
    if model.num_classes==1:
        labels = labels.float().to(device)
    else:
        labels = labels.long().to(device)
    pred = model(data)
    loss = loss_func(pred,labels)
    if training:
        loss.backward()
        optimizer.step()

    return loss, pred


def train_model(FLAGS,train_loader,val_loader,model,optimizer,loss_func,writer):
    # training
    best_train_loss = 10000
    best_val_loss = 10000
    best_model = copy.deepcopy(model)
    for ep in range(FLAGS.end_epoch):
        train_loss = 0
        for data,_,_,labels in train_loader:
            loss,_ = process_batch(FLAGS.device,data[:,-1],labels,model,optimizer,loss_func)
            train_loss += loss

        print(f'Epoch {ep} - Loss: {train_loss/len(train_loader)}')
        writer.add_scalar('Loss',train_loss.cpu()/len(train_loader),ep)

        #val check
        #torch.save(classifier.state_dict(), os.path.join(save_dir,f'e{ep}'))
        val_loss,_,_,_,_ = eval_model(FLAGS,val_loader,model,loss_func)
        #print(f'---- Val Acc = {val_acc}')
        writer.add_scalar('Val loss',val_loss,ep)
        if val_loss < best_val_loss:
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
    return best_train_loss / len(train_loader), \
            best_val_loss / len(val_loader), \
            best_model


def eval_model(FLAGS,loader,model,loss_func):
    with torch.no_grad():
        total_loss = 0
        all_predictions=[]
        all_labels=[]
        all_actions=[]
        all_objects=[]
        for data,actions,objects,labels in loader:
            loss,pred = process_batch(FLAGS.device,data[:,-1],labels,model,
                                      None,loss_func,False)
            total_loss += loss
            all_predictions.append(pred)
            all_labels.append(labels)
            all_actions.append(actions)
            all_objects.append(objects)
        all_actions = torch.cat(all_actions)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_objects = torch.cat(all_objects)

    return total_loss / len(loader), all_predictions.cpu(), all_labels.cpu(), all_actions.cpu(), all_objects

def test_model(device,loader,model,loss_func):
    with torch.no_grad():

        losses = 0
        all_predictions=[]
        all_labels=[]
        all_actions=[]
        all_objects=[]
        for data,actions,objects,labels in loader:
            all_objects.append(objects)
            all_actions.append(actions)
            all_labels.append(labels)

            loss,pred = process_batch(device,data[:,-1],labels,model,
                                      None,loss_func,False)
            #batch_distribs.append(sfmx(classifier_pred))
            if model.num_classes>1:
                _, pred = torch.max(pred, 1)
            #classifier_accuracy = (classifier_pred.data == l1).sum().item() / FLAGS.batch_size
            #accuracy[i] += classifier_accuracy
            losses += loss

            all_predictions.append(pred)
    #pdb.set_trace()
    all_actions= torch.cat(all_actions)
    all_labels=torch.cat(all_labels)
    all_predictions=torch.cat(all_predictions)
    all_objects = torch.cat(all_objects)
    #distribs = torch.cat(distribs)
    #entropy = torch.special.entr(distribs).sum(2).mean(dim=0)
    #accuracy /= len(loader)
    return losses / len(loader), all_predictions.cpu(), all_labels.cpu(), all_actions.cpu(), all_objects


def evaluate_all_models(model_dict,tr_ds,vl_ds,ts_ds):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(tr_ds,batch_size=128)
    val_loader = DataLoader(vl_ds,batch_size=128)
    test_loader = DataLoader(ts_ds,batch_size=128)
    #setup datasets
    all_results = []
    for prop,model in model_dict.items():
        prop_dict_list=[]
        for (set_nm,ds,load) in zip(['train','val','test'],
                                    [tr_ds,vl_ds,ts_ds],
                                    [train_loader,val_loader,test_loader]):
            ds.set_lbl(prop)
            loss,preds,lbls,acts,objs = test_model(device,load,model,nn.MSELoss())
            temp_dict = {'object':objs.tolist(),f'{prop}_lbls':lbls.tolist(),
                         'set_name':[set_nm]*len(lbls)}

            temp_dict[f'action']=acts.tolist()
            temp_dict[f'{prop}_pred']=preds.tolist()
            prop_dict_list.append(pd.DataFrame(temp_dict))

        all_results.append( pd.concat(prop_dict_list, ignore_index=True) )
    ar_df = pd.concat(all_results,axis=1)
    ar_df = ar_df.loc[:,~ar_df.columns.duplicated()]
    return ar_df

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--save_dir_id', type=str, default=0)
parser.add_argument('--batch_size', type=float, default=32)
parser.add_argument('--initial_learning_rate', type=float, default=0.00003)
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--weight_decay', type=float, default=1, help="weight decay for adam")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")
parser.add_argument('--dataset',type=str, default='new')
parser.add_argument('--test_properties', type=str, nargs='*', default=['pr_size','sq_size','stiffness','mass','contents_binary'])
parser.add_argument('--action_select', type=str, default=None, choices=['squeeze', 'press', 'shake', 'slide'])

parser.add_argument('--in_channels', type=int, default=195)
parser.add_argument('--hidden_dims', type=int, nargs='*', default=[128,128,128,128]) #[128,128,128])
parser.add_argument('--kernels', type=int, nargs='*', default=[6,6,5,4]) #[6,6,6])
parser.add_argument('--strides', type=int, nargs='*', default=[4,4,2,2]) #[3,3,3])
parser.add_argument('--paddings', type=int, nargs='*', default=[1,1,1,0]) #[1,1,2])

#If not splitting by object, can also have 'ball_id','contents_fine'

class_FLAGS = parser.parse_args()

if __name__ == '__main__':
    save_path = f'baseline_results/{class_FLAGS.save_dir_id}_{class_FLAGS.action_select}'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    #pdb.set_trace()
    print('setting up model dataset')
    td,vd,sd,properties = setup_dataset(class_FLAGS)
    properties.to_csv(os.path.join(save_path,'properties.csv'))
    with open(os.path.join(save_path,'config.yaml'), 'w') as conf_file:
        yaml.dump(class_FLAGS, conf_file)

    prop_models = {}
    for prop in class_FLAGS.test_properties:
        print(f'Classify by {prop}')
        setattr(class_FLAGS,'label',prop)
        mod = run_all(class_FLAGS,save_path,td,vd,sd,prop)
        if 'contents' not in prop:
            prop_models[prop] = mod

    all_df = evaluate_all_models(prop_models,td,vd,sd)
    lbls=[]
    for i in range(len(all_df)):
        x = np.array(all_df.iloc[i][['pr_size_pred','sq_size_pred','stiffness_pred','mass_pred']])
        lbls.append(((properties[['pr_size','sq_size','stiffness','mass']]-x)**2).sum(axis=1).idxmin())
    all_df['object_pred']=lbls
    all_df.to_pickle(os.path.join(save_path,'all_results'))
