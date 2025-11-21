import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import time
from tempfile import TemporaryDirectory
import random
import math
import yaml
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, utils
from torchvision.io import read_image
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset, random_split, RandomSampler
from torcheval.metrics import MulticlassPrecisionRecallCurve
from torcheval.metrics.functional import binary_auprc
import timm
from torchvision.models.video import r2plus1d_18

from sklearn import metrics as skmet
from sklearn.model_selection import StratifiedKFold

# Local
from utils import pad, num_parameters, EarlyStopper, custom_collate
from datasets import MILMRI, MRI
from models import MILNet, StandardMILNet
import transforms

np.random.seed(1234)
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
--------------------------
*** TRAINING FUNCTIONS ***
--------------------------
'''
def train_one_epoch(model, optimizer, scheduler, train_loader, device, cfg):
    model.train()

    num_steps_per_epoch = len(train_loader)

    target_ls = []
    output_ls = []
    losses = []
    for i, batch in enumerate(train_loader):
        inputs,targets = batch
        inputs = inputs.to(device).float()
        targets = targets.to(device).type(torch.float32)
        outputs = model(inputs)
        output_ls.append(outputs.cpu().detach().numpy())
        target_ls.append(targets.cpu().detach().numpy())
        loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), targets.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
    scheduler.step()
    
    metrics = compute_metrics_bin(np.concatenate(target_ls), np.concatenate(output_ls), cfg)
    return np.mean(losses), metrics 

def train_one_epoch_mil(model, optimizer, scheduler, train_loader, device, cfg):
    model.train()

    num_steps_per_epoch = len(train_loader)

    target_ls = []
    output_ls = []
    losses = []
    for i, batch in enumerate(train_loader):
        inputs = batch['study'].to(device).type(torch.float32).unsqueeze(1)
        num_imgs = batch['num_imgs']
        targets = batch['label'].to(device).type(torch.float32)
        target_ls.append(targets.cpu().detach().numpy())
        #print(inputs.shape)
        #print(targets.shape)
        #print(len(num_imgs))
        outputs, _ = model(inputs, num_imgs)
        output_ls.append(outputs.cpu().detach().numpy())
        #print(outputs.shape)
        #print(outputs)
        if cfg['net_kwargs']['pooling_method'] == 'stan':
            loss = nn.BCELoss()(outputs.squeeze(), targets.squeeze())
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), targets.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        #print(f"\tBatch {i+1} of {num_steps_per_epoch}. Loss={loss.detach().item():0.3f}", end='\r')
    scheduler.step()
    
    #print(' '*100, end='\r')
    
    metrics = compute_metrics_bin(np.concatenate(target_ls), np.concatenate(output_ls), cfg)
    return np.mean(losses), metrics 

def evaluate(model, test_loader, device, cfg):
    model.eval()

    num_steps_per_epoch = len(test_loader)

    target_ls = []
    output_ls = []
    study_ls = []
    losses = []
    for i, batch in enumerate(test_loader):
        inputs,targets = batch
        inputs = inputs.to(device).float()
        targets = targets.to(device).type(torch.float32)
        target_ls.append(targets.cpu().numpy())
        
        with torch.no_grad():
            outputs = model(inputs)
            output_ls.append(outputs.cpu().numpy())
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), targets.squeeze())
            
        losses.append(loss.detach().item())
        
    metrics = compute_metrics_bin(np.concatenate(target_ls), np.concatenate(output_ls), cfg)
    return np.mean(losses), metrics

def evaluate_mil(model, test_loader, device, cfg):
    model.eval()

    num_steps_per_epoch = len(test_loader)

    target_ls = []
    output_ls = []
    losses = []
    for i, batch in enumerate(test_loader):
        inputs = batch['study'].to(device).type(torch.float32).unsqueeze(1)
        num_imgs = batch['num_imgs']
        targets = batch['label'].to(device).type(torch.float32)
        target_ls.append(targets.cpu().numpy())
        
        with torch.no_grad():
            outputs, _ = model(inputs, num_imgs)
            output_ls.append(outputs.cpu().numpy())
            if cfg['net_kwargs']['pooling_method'] == 'stan':
                loss = nn.BCELoss()(outputs.squeeze(), targets.squeeze())
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), targets.squeeze())
            
        losses.append(loss.detach().item())
        
    metrics = compute_metrics_bin(np.concatenate(target_ls), np.concatenate(output_ls), cfg)
    return np.mean(losses), metrics

def compute_metrics_bin(y_true, y_pred, cfg):
    mets = dict()
    
    if cfg['net_kwargs']['pooling_method'] != 'stan':
        y_pred = 1/(1+np.exp(-y_pred)) #sigmoid to get probability 0-1
    y_pred_cls = np.rint(y_pred)
    
    mets['roc_auc'] = skmet.roc_auc_score(y_true, y_pred)
    mets['conf_mat'] = skmet.confusion_matrix(y_true, y_pred_cls)
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['precision'] = skmet.precision_score(y_true, y_pred_cls,zero_division=0)
    mets['sensitivity'] = skmet.recall_score(y_true, y_pred_cls,zero_division=0)
    mets['specificity'] = skmet.recall_score(y_true, y_pred_cls, pos_label=0,zero_division=0)
    mets['f1score'] = skmet.f1_score(y_true, y_pred_cls,zero_division=0)
    mets['y_pred'] = y_pred
    mets['y_true'] = y_true
    
    return mets

def compute_metrics_multi(y_true, y_pred):
    mets = dict()
    
    y_pred = torch.softmax(torch.tensor(y_pred), dim=1).squeeze().numpy() #softmax to get probability 0-1
    y_pred_cls = np.argmax(y_pred, axis=1)
    
    mets['roc_auc'] = skmet.roc_auc_score(y_true, y_pred, multi_class='ovr')
    mets['conf_mat'] = skmet.confusion_matrix(y_true, y_pred_cls)
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['precision'] = skmet.precision_score(y_true, y_pred_cls, average='macro', zero_division=0)
    mets['sensitivity'] = skmet.recall_score(y_true, y_pred_cls, average='macro', zero_division=0)
    #mets['specificity'] = skmet.recall_score(y_true, y_pred_cls, average='macro',zero_division=0)
    mets['f1score'] = skmet.f1_score(y_true, y_pred_cls, average='macro', zero_division=0)
    mets['y_pred'] = y_pred
    mets['y_true'] = y_true
    
    return mets


def main(cfg):
    
    os.makedirs(cfg['save_dir'], exist_ok=True)
    
    # copy the config file to the artifact folder
    with open(cfg['save_dir'] + '/config.yaml', 'w') as f: 
        yaml.dump(cfg, f)
        
    # Read in metadata file
    imgs_df = pd.read_csv(cfg['metadata_csv'])
    # balancing
    if cfg['balance'] != 0:
        healthy_df = imgs_df[imgs_df['ClinSig'] == 0]
        unhealthy_df = imgs_df[imgs_df['ClinSig'] == 1]
        healthy_df = healthy_df.loc[healthy_df['ProxID'].isin(healthy_df['ProxID'].drop_duplicates().sample(frac=cfg['balance'],random_state=123, ignore_index=True))]
        imgs_df = pd.concat([healthy_df,unhealthy_df])
    
    # Cross Validation
    skf = StratifiedKFold(n_splits=cfg['folds'], shuffle=True, random_state=123)
    patient_df = imgs_df.drop_duplicates(subset=['ProxID'], ignore_index=True, keep='last')
    for i, (train_index, val_index) in enumerate(skf.split(patient_df['ProxID'], patient_df['ClinSig'])):

        print(f"----------\nFOLD {i+1}:\n----------")
        
        # Model
        if not cfg['mil']:
            model = timm.create_model(**cfg['encoder_kwargs'])
            model = model.to(device)
        else: # if mil
            encoder = timm.create_model(**cfg['encoder_kwargs'])
            num_features = list(encoder.children())[-1].in_features #num_features = encoder.fc.in_features
            encoder = nn.Sequential(*list(encoder.children())[:-1])
            num_pars_encoder = num_parameters(encoder)
            #if cfg['dp']:
            #    milenc = MILNet(encoder, num_features=num_features, **cfg['net_kwargs'], pooling_method=cfg['inst_pooling'])
            #    model = MILNet(milenc, num_features=num_features, **cfg['net_kwargs'], tier=2, pooling_method=cfg['bag_pooling'])
            #else:
            if cfg['net_kwargs']['pooling_method'] == 'stan':
                model = StandardMILNet(encoder, num_features=num_features, dropout=cfg['net_kwargs']['drop_rate'], debug=cfg['net_kwargs']['debug'])
            else:
                model = MILNet(encoder, num_features=num_features, **cfg['net_kwargs'])
            model = model.to(device)
            if cfg['pretrain_enc']:
                # load pretrained encoder
                checkpoint = torch.load(cfg['pretrained_model_dir']+f'/model_fold{i+1}.ckpt', weights_only=False)['model_state_dict']
                checkpoint = OrderedDict(zip(encoder.state_dict().keys(),list(checkpoint.values())[:-2])) # remove last fc weight and bias
                model.encoder.load_state_dict(checkpoint) # load weights
            if cfg['pretrain_fc']:
                # load pretrained fc layer
                checkpoint = torch.load(cfg['pretrained_model_dir']+f'/model_fold{i+1}.ckpt', weights_only=False)['model_state_dict']
                with torch.no_grad():
                    model.fc[1].weight.copy_(list(checkpoint.items())[-2][1])
                    model.fc[1].bias.copy_(list(checkpoint.items())[-1][1])
            if cfg['freeze_enc']:
                # Freezing Encoder
                for param in model.encoder.parameters():
                    param.requires_grad = False
            if cfg['freeze_fc']:
                # Freezing FC Layer
                for param in model.fc.parameters():
                    param.requires_grad = False
        num_pars = num_parameters(model)

        # Datasets
        train_patients = patient_df.iloc[train_index]['ProxID']
        val_patients = patient_df.iloc[val_index]['ProxID']
        train_df = imgs_df.loc[imgs_df['ProxID'].isin(train_patients)]
        val_df = imgs_df.loc[imgs_df['ProxID'].isin(val_patients)]
        view_filter = cfg['view_filter'] #['t2_tse_cor','t2_tse_sag','t2_tse_tra']
        if cfg['mil']:
            train_dataset = MILMRI(train_df, data_path=cfg['data_dir'], transforms=transforms.transform_augment, view_filter=view_filter, save_dir=cfg['save_dir']+f'/train_fold{i+1}.csv', use_k=cfg['use_k'])
            val_dataset = MILMRI(val_df, data_path=cfg['data_dir'], transforms=transforms.transform_original, view_filter=view_filter, save_dir=cfg['save_dir']+f'/val_fold{i+1}.csv', use_k=cfg['use_k'])
            collate_fn = custom_collate
        else:
            train_dataset = MRI(train_df, data_path=cfg['data_dir'], transforms=transforms.transform_augment, view_filter=view_filter, save_dir=cfg['save_dir']+f'/train_fold{i+1}.csv', use_k=cfg['use_k'])
            val_dataset = MRI(val_df, data_path=cfg['data_dir'], transforms=transforms.transform_original, view_filter=view_filter, save_dir=cfg['save_dir']+f'/val_fold{i+1}.csv', use_k=cfg['use_k'])
            collate_fn = default_collate
        
        # Dataloaders
        train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, **cfg['loader_kwargs'])
        val_loader = DataLoader(val_dataset, shuffle=True, collate_fn=collate_fn, **cfg['loader_kwargs'])

        # Hyperparameters and arguments
        optimizer = optim.Adam(model.parameters(),**cfg['optim_kwargs'])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **cfg['sched_kwargs'])

        train_loss_ls = []
        val_loss_ls = []
        train_acc_ls = []
        val_acc_ls = []
        train_prec_ls = []
        val_prec_ls = []
        train_sens_ls = []
        val_sens_ls = []
        #train_spec_ls = []
        #val_spec_ls = []
        train_rocauc_ls = []
        val_rocauc_ls = []
        #train_prcauc_ls = []
        #val_prcauc_ls = []
        train_f1_ls = []
        val_f1_ls = []
        train_pred_ls = []
        val_pred_ls = []
        train_true_ls = []
        val_true_ls = []
        best_loss = 100
        criterion = nn.CrossEntropyLoss()

        epochs = cfg['num_epochs']
        verbose = cfg['verbose']
        since = time.time()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}\n","-"*10) if verbose else None

            # train for a single epoch
            if cfg['mil']:
                train_loss, train_metrics = train_one_epoch_mil(model, optimizer, lr_scheduler, train_loader, device, cfg)
            else:
                train_loss, train_metrics = train_one_epoch(model, optimizer, lr_scheduler, train_loader, device, cfg)
            train_loss_ls.append(train_loss)
            train_acc_ls.append(train_metrics['accuracy'])
            train_prec_ls.append(train_metrics['precision'])
            train_sens_ls.append(train_metrics['sensitivity'])
            #train_spec_ls.append(train_metrics['specificity'])
            train_rocauc_ls.append(train_metrics['roc_auc'])
            #train_prcauc_ls.append(train_metrics['prc_auc'])
            train_f1_ls.append(train_metrics['f1score'])
            train_pred_ls.append(train_metrics['y_pred'])
            train_true_ls.append(train_metrics['y_true'])
            print(f"[TRAIN] BCE loss: {train_loss:0.4f} | Acc: {100*train_metrics['accuracy']:0.2f}%") if verbose else None

            # evaluate
            if cfg['mil']:
                val_loss, val_metrics = evaluate_mil(model, val_loader, device, cfg)
            else:
                val_loss, val_metrics = evaluate(model, val_loader, device, cfg)
            val_loss_ls.append(val_loss)
            val_acc_ls.append(val_metrics['accuracy'])
            val_prec_ls.append(val_metrics['precision'])
            val_sens_ls.append(val_metrics['sensitivity'])
            #val_spec_ls.append(val_metrics['specificity'])
            val_rocauc_ls.append(val_metrics['roc_auc'])
            #val_prcauc_ls.append(val_metrics['prc_auc'])
            val_f1_ls.append(val_metrics['f1score'])
            val_pred_ls.append(val_metrics['y_pred'])
            val_true_ls.append(val_metrics['y_true'])
            
            print(f"[VALID] BCE loss: {val_loss:0.4f} | Acc: {100*val_metrics['accuracy']:0.2f}%") if verbose else None
            print() if verbose else None
            if not verbose and (epoch % int(epochs*.25) == 0):
                print(f"Epoch {epoch}/{epochs}")

        print("Finished Training.")
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        torch.save({'epoch':epoch, 
                    'model_state_dict':model.state_dict(), 
                    #'optimizer_state_dict':optimizer.state_dict(),
                    'loss':val_loss, 
                    'metrics':val_metrics}, 
                    cfg['save_dir']+f'/model_fold{i+1}.ckpt')
        checkpoint = torch.load(cfg['save_dir']+f'/model_fold{i+1}.ckpt', weights_only=False)
        print(f"Best Validation Epoch: {checkpoint['epoch']},\n\tLoss: {checkpoint['loss']:.4f},\n\tAccuracy: {checkpoint['metrics']['accuracy']*100:.2f},\n\tAUROC: {checkpoint['metrics']['roc_auc']*100:.2f}")
        
        # Metrics excel file
        metrics_df = pd.DataFrame({'Epoch':range(0,epochs), 
                                   'Training Loss':train_loss_ls, 'Validation Loss':val_loss_ls,
                                   'Training Accuracy':train_acc_ls, 'Validation Accuracy':val_acc_ls,
                                   'Training Precision':train_prec_ls, 'Validation Precision':val_prec_ls,
                                   'Training Sensitivity':train_sens_ls, 'Validation Sensitivity':val_sens_ls,
                                   #'Training Specificity':train_spec_ls, 'Validation Specificity':val_spec_ls,
                                   'Training F1 Score':train_f1_ls, 'Validation F1 Score':val_f1_ls,
                                   'Training ROCAUC':train_rocauc_ls, 'Validation ROCAUC':val_rocauc_ls,
                                   #'Training PRCAUC':train_prcauc_ls, 'Validation PRCAUC':val_prcauc_ls,
                                  })
        metrics_df.to_csv(cfg['save_dir']+f'/metrics_fold{i+1}.csv')
        # Predictions at last epoch csv file
        preds_df = pd.DataFrame({'Validation Pred':val_pred_ls[-1].squeeze(), 'Validation True':val_true_ls[-1].squeeze()})
        preds_df.to_csv(cfg['save_dir']+f'/preds_fold{i+1}.csv')



if __name__=='__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser('Train Shoulder X-ray classifier')
    parser.add_argument('--config', type=str, required=False, default=None,
                        help='YAML configuration file.')
    parser.add_argument('--num-heads', type=int, required=False, default=None,
                        help='Number of attention heads for MedVidNet model. Overrides "net_kwargs: num_heads" in config if provided.')
    parser.add_argument('--pooling-method', type=str, required=False, default=None,
                        choices=['attn', 'tanh_attn', 'max', 'avg'],
                        help='Pooling method. Overrides "vidnet_kwargs: pooling_method" in config if provided.')
    parser.add_argument('--device', type=str, required=False, default=None,
                        help='Compute devie. Overrides "device" in config if provided')
    args = parser.parse_args()
    
    # load config
    if args.config != None:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        with open(snakemake.input.config, 'r') as f:
            cfg = yaml.safe_load(f)
    
        
        
    print("Running training script with configuration:")
    print('-'*30)
    yaml.dump(cfg, sys.stdout)
    print('-'*30)

    main(cfg)


