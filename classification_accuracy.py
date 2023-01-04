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
from networks import Encoder, Decoder, Property_model
from utils import weights_init, reparameterize, setup_models, cNs_init
from ball_datasets import tactile_explorations,split_indices,latent_representations
import training
from property_maps import property_df,get_num_classes
import pandas as pd

from tensorboardX import SummaryWriter

import pdb

# Set up dataset so don't have to rerun
def setup_dataset(FLAGS):
    ds_train = tactile_explorations(train=True,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    ds_val = tactile_explorations(train=True,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    ds_test = tactile_explorations(train=False,dataset=FLAGS.dataset,action_select=FLAGS.action_select)
    train_indices, val_indices = split_indices(ds_train,4,FLAGS.dataset)
    ds_train.set_indices(train_indices)
    ds_val.set_indices(val_indices)
    ds_train.set_transform()
    ds_val.set_transform(ds_train.get_transform())
    ds_test.set_transform(ds_train.get_transform())
    props = property_df('cluster')
    return ds_train,ds_val,ds_test,props
# ds_train,ds_val,ds_test,properties = setup_dataset()


def gen_latent_dataset(FLAGS,loader,encoder,decoder,action_names,device):
    latent = []
    with torch.no_grad():
        for it, (X, action_batch, obj_batch) in enumerate(loader):
            batch = pd.DataFrame(index=np.arange(X.shape[0]), columns=['cm','clv','sm','slv','actions','object'])
            X = X.to(device)
            action_batch = action_batch.to(device)
            context,style_mu,style_logvar = cNs_init(X.size(0),X.size(1),
                                                     FLAGS.class_dim,FLAGS.style_dim,device)
            cm_ar,clv_ar,sm_ar,slv_ar = [],[],[],[]
            for i in range(4):
                sm,slv,cm,clv,_,_=training.single_pass(X, action_batch, i, context,
                                                       style_mu, style_logvar,
                                                       encoder, decoder, training=True)
                #print(training.loss(FLAGS,cm,clv,sm,slv,mu_x,lv_x,X))
                cm_ar.append(cm.cpu())
                clv_ar.append(clv.cpu())
                sm_ar.append(sm.cpu())
                slv_ar.append(slv.cpu())
                context = torch.cat([cm,clv],dim=1)

            cm_ar = torch.stack(cm_ar,dim=1)
            clv_ar = torch.stack(clv_ar,dim=1)
            sm_ar = torch.stack(sm_ar,dim=1)
            slv_ar = torch.stack(slv_ar,dim=1)
            actions = torch.flip(action_batch,[1]).cpu()
            act_batch_names = action_names[torch.where(actions==1)[2]].reshape([-1,X.shape[1]])
            for i in range(X.shape[0]):
                row = batch.iloc[i]
                row['cm'] = cm_ar[i]
                row['clv'] = clv_ar[i]
                row['sm'] = sm_ar[i]
                row['slv'] = slv_ar[i]
                row['actions'] = act_batch_names[i]
                row['object'] = obj_batch[i].item()
                batch.iloc[i] = row

            latent.append(batch)
    return pd.concat(latent, axis=0, ignore_index=True)

def create_datasets(run_folder,ds_train=None,ds_val=None,ds_test=None,properties=None,multiplier=1):
    encoder,decoder,sgvae_FLAGS,device = setup_models(run_folder)
    sgvae_FLAGS.batch_size = 128
    if ds_train is None:
        ds_train,ds_val,ds_test,properties = setup_dataset()
    action_names = ds_train.df['action'].unique()

    train_df = []
    for i in range(multiplier):
        ds_train.random_context_sampler()
        train_loader = DataLoader(ds_train,batch_size=sgvae_FLAGS.batch_size)

        train_df.append(gen_latent_dataset(sgvae_FLAGS, train_loader, encoder, decoder, action_names, device))
    train_df = pd.concat(train_df)

    ds_val.random_context_sampler()
    ds_test.random_context_sampler()


    val_loader = DataLoader(ds_val,batch_size=sgvae_FLAGS.batch_size)
    test_loader = DataLoader(ds_test,batch_size=sgvae_FLAGS.batch_size)

    val_df = gen_latent_dataset(sgvae_FLAGS, val_loader, encoder, decoder, action_names, device)
    test_df = gen_latent_dataset(sgvae_FLAGS, test_loader, encoder, decoder, action_names, device)

    train_ds = latent_representations(train_df,properties)
    val_ds = latent_representations(val_df,properties)
    test_ds = latent_representations(test_df,properties)

    return train_ds,val_ds,test_ds


def classify_latent_vectors(FLAGS,log_dir,tr_ds,vl_ds,ts_ds,prop,latent_types=['cm','sm']):
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

    for latent_type in latent_types:
        mod_type = 'content' if latent_type=='cm' else 'style'
        dim=tr_ds.data[latent_type].iloc[0].shape[1]
        model = Property_model(z_dim=dim, num_classes=class_cnt).to(FLAGS.device)
        model.apply(weights_init)
        optimizer = optim.Adam(
            list(model.parameters()),
            lr=FLAGS.initial_learning_rate,
            betas=(FLAGS.beta_1, FLAGS.beta_2),
            weight_decay=FLAGS.weight_decay
        )

        train_loader = DataLoader(tr_ds,batch_size=FLAGS.batch_size,shuffle=True)
        val_loader = DataLoader(vl_ds,batch_size=FLAGS.batch_size,shuffle=True)
        test_loader = DataLoader(ts_ds,batch_size=FLAGS.batch_size,shuffle=True)

        writer = SummaryWriter(log_dir=f'{log_dir}/{latent_type}')
        tr_ds.set_classifier(latent_type[0])
        vl_ds.set_classifier(latent_type[0])
        ts_ds.set_classifier(latent_type[0])

        _,_,best_model = train_model(FLAGS,train_loader,val_loader,
                                     model,optimizer,loss_func,writer)
        writer.close()
        torch.save(best_model.state_dict(),os.path.join(log_dir,f'{mod_type}_model'))
        load_dict = {'train':train_loader,'val':val_loader,
                     'test':test_loader}
        for set_id,load in load_dict.items():
            loss,preds,lbls,acts,objs = test_model(FLAGS.device,load,best_model,
                                                   loss_func)
            save_dict = {f'{set_id}_loss':loss,f'{set_id}_preds':preds,
                         f'{set_id}_lbls':lbls,f'{set_id}_actions':acts,
                         f'{set_id}_objs':objs}
            torch.save(save_dict, os.path.join(log_dir,f'{mod_type}_dict'))

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
        for data,actions,objs,labels in train_loader:
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

        losses = np.zeros(4)
        all_predictions=[]
        all_labels=[]
        all_actions=[]
        all_objects=[]
        for data,actions,objects,labels in loader:
            all_objects.append(objects)
            all_actions.append(actions)
            all_labels.append(labels)
            batch_preds = []
            for i in range(data.shape[1]):
                loss,pred = process_batch(device,data[:,i],labels,model,
                                          None,loss_func,False)
                #batch_distribs.append(sfmx(classifier_pred))
                if model.num_classes>1:
                    _, pred = torch.max(pred, 1)
                #classifier_accuracy = (classifier_pred.data == l1).sum().item() / FLAGS.batch_size
                #accuracy[i] += classifier_accuracy
                batch_preds.append(pred)
                losses[i] += loss

            all_predictions.append(torch.stack(batch_preds).T)

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
            for i in range(acts.shape[1]):
                temp_dict[f'action_seq{i+1}']=acts[:,i].tolist()
                temp_dict[f'{prop}_pred_seq{i+1}']=preds[:,i].tolist()
            prop_dict_list.append(pd.DataFrame(temp_dict))
        all_results.append( pd.concat(prop_dict_list, ignore_index=True) )
    ar_df = pd.concat(all_results)
    ar_df = ar_df.loc[:,~ar_df.columns.duplicated()]

    return ar_df

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--save_dir_id', type=str, default=0)
parser.add_argument('--vae_model', type=str, default='Aug25_09-46-47_g04712288959-33')
parser.add_argument('--batch_size', type=float, default=32)
parser.add_argument('--initial_learning_rate', type=float, default=0.00001)
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--weight_decay', type=float, default=1, help="weight decay for adam")
parser.add_argument('--end_epoch', type=int, default=1, help="flag to indicate the final epoch of training")
parser.add_argument('--train_multiplier',type=int, default=5, help="How many random random sequences to generate for training")
parser.add_argument('--dataset',type=str, default='new')
parser.add_argument('--test_properties', type=str, nargs='*', default=['pr_size','sq_size','stiffness','mass','contents_binary'])
parser.add_argument('--action_select', type=str, default=None, choices=['squeeze', 'press', 'shake', 'slide'])

#If not splitting by object, can also have 'ball_id','contents_fine'

class_FLAGS = parser.parse_args()

if __name__ == '__main__':
    dire = 'runs'
    file = class_FLAGS.vae_model
    load_path = os.path.join(dire,file)
    save_path = f'classification_results/{file}_{class_FLAGS.save_dir_id}'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    #pdb.set_trace()
    print('setting up model dataset')
    ds_train,ds_val,ds_test,properties = setup_dataset(class_FLAGS)
    properties.to_csv(os.path.join(save_path,'properties.csv'))
    with open(os.path.join(save_path,'config.yaml'), 'w') as conf_file:
        yaml.dump(class_FLAGS, conf_file)
    print('generating latent representation dataset')
    td,vd,sd = create_datasets(load_path,ds_train,ds_val,ds_test,properties,class_FLAGS.train_multiplier)

    prop_models = {}
    for prop in class_FLAGS.test_properties:
        print(f'Classify by {prop}')
        setattr(class_FLAGS,'label',prop)
        mod = classify_latent_vectors(class_FLAGS,save_path,td,vd,sd,prop)
        if 'contents' not in prop:
            prop_models[prop] = mod
    all_df = evaluate_all_models(prop_models,td,vd,sd)
    all_df.to_pickle(os.path.join(save_path,'all_results'))


    
