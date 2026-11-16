import sys, os
# shared helpers (transforms.py, utils.py, radimagenet.py) live one folder up
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import timm
from timm import optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics as skmet
import os
import json
import yaml
import copy
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import time

# locally defined
import transforms as my_transforms
from dataset import VideoDataset, collate_video_batch
import models
from radimagenet import load_radimagenet_resnet50
from utils import num_parameters, EarlyStopper
from sklearn.calibration import calibration_curve

# per-run output dir, set in main(); used as default for plots
SAVE_DIR = None

#set rng seeds
torch.manual_seed(2)
np.random.seed(2)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def train_one_epoch(model, optimizer, train_dataloader, device, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()
    scaler = GradScaler()
    
    losses = []
    for ix, batch in enumerate(train_dataloader):
        inputs = batch['video'].to(device)
        num_frames = batch['num_frames']
        study_indices = batch['study_indices'].to(device)
        targets = batch['quality_label'].to(device).type(torch.float32)
        
        with autocast():
            outputs, _ = model(inputs, num_frames, study_indices)
            
            # Group targets by study
            unique_studies = torch.unique(study_indices)
            study_targets = []
            for study_idx in unique_studies:
                study_mask = (study_indices == study_idx)
                study_target = targets[study_mask].float().mean()  # average targets for same study
                study_targets.append(study_target)
            study_targets = torch.stack(study_targets)  # [num_unique_studies]
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs.squeeze(), study_targets)
            # loss = nn.functional.binary_cross_entropy_with_logits(
            #     outputs.squeeze(), targets.squeeze())
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (ix + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        losses.append(loss.detach().item() * accumulation_steps)
        
    return np.mean(losses)
            
def evaluate(model, test_dataloader, device, cv_index=None):
    model.eval()
    target_ls = []
    output_ls = []
    video_name_ls = []
    patient_id_ls = []
    losses = []
    
    for batch in test_dataloader:
        inputs = batch['video'].to(device)
        num_frames = batch['video_lengths']
        study_indices = batch['study_indices'].to(device)
        targets = batch['quality_label'].to(device).type(torch.float32)
        video_names = batch['video_name']
        patient_ids = batch['patient_id']
        
        with torch.no_grad(), autocast():
            outputs, _ = model(inputs, num_frames, study_indices)
            # Group targets by study
            unique_studies = torch.unique(study_indices)
            study_targets = []
            study_video_names = []
            study_patient_ids = []
            
            for study_idx in unique_studies:
                study_mask = (study_indices == study_idx)
                study_target = targets[study_mask].float().mean()
                study_targets.append(study_target)
                
                # Take first video name and patient id for each study
                mask_indices = study_mask.nonzero().squeeze()
                first_idx = mask_indices[0] if mask_indices.dim() > 0 else mask_indices
                study_video_names.append(video_names[first_idx])
                study_patient_ids.append(patient_ids[first_idx])
            
            study_targets = torch.stack(study_targets)
            
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs.squeeze(), study_targets)
            
            # Store study-level results
            target_ls.append(study_targets.cpu().numpy())
            output_ls.append(outputs.cpu().numpy())
            video_name_ls.append(study_video_names)
            patient_id_ls.append(study_patient_ids)
            losses.append(loss.item())
    
    # Concatenate results
    metrics = compute_metrics(
        np.concatenate(target_ls),
        np.concatenate(output_ls),
        cv_index=cv_index
    )
    
    return (np.mean(losses), metrics, 
            np.concatenate(target_ls), 
            np.concatenate(output_ls),
            np.concatenate(video_name_ls), 
            np.concatenate(patient_id_ls))

def compute_calibration_plot(y_true, y_pred, cv_index=None, from_probs=False, n_bins=10, 
                           save_dir=None):
    """
    Compute and plot the calibration curve for the model predictions.
    
    Args:
        y_true (np.ndarray): Ground truth binary labels
        y_pred (np.ndarray): Model predictions (logits or probabilities)
        cv_index (int, optional): Cross-validation fold index
        from_probs (bool): Whether y_pred is already in probability space
        n_bins (int): Number of bins for calibration curve
        save_dir (str): Directory to save the calibration plot
    """
    save_dir = save_dir or SAVE_DIR
    # Convert predictions to probabilities if needed
    if from_probs:
        y_pred_prob = np.clip(y_pred, 0, 1)  # New: clip the probabilities
    else:
        y_pred_prob = 1/(1+np.exp(-y_pred))
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true=y_true.squeeze(),
        y_prob=y_pred_prob.squeeze(),
        n_bins=n_bins,
        strategy='uniform'
    )
    
    # Create calibration plot
    plt.figure(figsize=(8, 8))
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Plot model calibration
    plt.plot(prob_pred, prob_true, 'o-', label='Model calibration')
    
    # Customize plot
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    if cv_index is not None:
        plt.savefig(f'{save_dir}/calibration_curve_cv{cv_index}.png')
    else:
        plt.savefig(f'{save_dir}/calibration_curve_aggregate.png')
    plt.close()
    
    # Compute calibration metrics
    # Calculate mean calibration error
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    return {
        'calibration_error': calibration_error,
        'true_probs': prob_true,
        'pred_probs': prob_pred
    }    
def compute_metrics(y_true, y_pred, cv_index=None, from_probs=False, threshold=0.5, save_dir=None):
    save_dir = save_dir or SAVE_DIR
    mets = dict()
    
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    
    # Convert logits to probabilities if needed
    if not from_probs:
        y_pred_prob = 1/(1+np.exp(-y_pred))
    else:
        y_pred_prob = y_pred
    # Get binary predictions
    y_pred_cls = (y_pred_prob > threshold).astype(int)
    
    # Compute metrics
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['sensitivity'] = skmet.recall_score(y_true, y_pred_cls)
    mets['specificity'] = skmet.recall_score(y_true, y_pred_cls, pos_label=0)
    mets['roc_auc'] = skmet.roc_auc_score(y_true, y_pred_prob)
    mets['balanced_accuracy'] = skmet.balanced_accuracy_score(y_true, y_pred_cls)
    mets['average_precision'] = skmet.average_precision_score(y_true, y_pred_prob)
    mets['f1'] = skmet.f1_score(y_true, y_pred_cls)
    mets['precision'] = skmet.precision_score(y_true, y_pred_cls)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curves for both classes
    colors = ['yellow', 'green']
    for i, class_name in enumerate(['noPDA', 'PDA']):
        # ROC curve for each class
        fpr, tpr, _ = skmet.roc_curve(y_true == i, y_pred_prob if i == 1 else 1 - y_pred_prob)
        roc_auc = skmet.roc_auc_score(y_true == i, y_pred_prob if i == 1 else 1 - y_pred_prob)
        ax1.plot(fpr, tpr, color=colors[i], label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--')  # diagonal line
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Precision-Recall curves for both classes
    for i, class_name in enumerate(['noPDA', 'PDA']):
        # PR curve for each class
        precision, recall, _ = skmet.precision_recall_curve(y_true == i, 
                                                          y_pred_prob if i == 1 else 1 - y_pred_prob)
        ap = skmet.average_precision_score(y_true == i, 
                                         y_pred_prob if i == 1 else 1 - y_pred_prob)
        ax2.plot(recall, precision, color=colors[i], label=f'{class_name} (AP = {ap:.3f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if cv_index is not None:
        plt.savefig(f'{save_dir}/metrics_curves_cv{cv_index}.png')
    else:
        plt.savefig(f'{save_dir}/metrics_curves_aggregate.png')
    plt.close()
    calibration_results = compute_calibration_plot(
        y_true=y_true,
        y_pred=y_pred,
        cv_index=cv_index,
        from_probs=from_probs,
        save_dir=save_dir
    )
    
    # Add calibration error to metrics
    mets['calibration_error'] = calibration_results['calibration_error']
    return mets

def aggregate_cv_results(all_y_true, all_y_pred_prob, all_metrics, save_dir=None):
    """
    Aggregate results from all CV folds
    
    Parameters:
    all_y_true: list of true labels from each fold
    all_y_pred_prob: list of prediction probabilities from each fold
    all_metrics: list of metric dictionaries from each fold
    """
    save_dir = save_dir or SAVE_DIR
    # Combine all predictions
    y_true_combined = np.concatenate(all_y_true)
    y_pred_prob_combined = np.concatenate(all_y_pred_prob)
    
    # Calculate mean and std of metrics across folds
    cv_summary = {metric: {
        'mean': np.mean([m[metric] for m in all_metrics]),
        'std': np.std([m[metric] for m in all_metrics])
    } for metric in all_metrics[0].keys()}
    
    # Print CV summary
    print("\nCross-validation Summary:")
    for metric in cv_summary:
        print(f"{metric}:")
        print(f"  Mean: {cv_summary[metric]['mean']:.3f} ± {cv_summary[metric]['std']:.3f}")
    
    # Generate plots for combined data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curves for both classes
    colors = ['blue', 'red']
    for i, class_name in enumerate(['noPDA', 'PDA']):
        fpr, tpr, _ = skmet.roc_curve(y_true_combined == i, 
                                     y_pred_prob_combined if i == 1 else 1 - y_pred_prob_combined)
        roc_auc = skmet.roc_auc_score(y_true_combined == i, 
                                     y_pred_prob_combined if i == 1 else 1 - y_pred_prob_combined)
        ax1.plot(fpr, tpr, color=colors[i], label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Aggregate ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Precision-Recall curves for both classes
    for i, class_name in enumerate(['noPDA', 'PDA']):
        precision, recall, _ = skmet.precision_recall_curve(y_true_combined == i, 
                                                          y_pred_prob_combined if i == 1 else 1 - y_pred_prob_combined)
        ap = skmet.average_precision_score(y_true_combined == i, 
                                         y_pred_prob_combined if i == 1 else 1 - y_pred_prob_combined)
        ax2.plot(recall, precision, color=colors[i], label=f'{class_name} (AP = {ap:.3f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Aggregate Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_curves_aggregate.png')
    plt.close()
    calibration_results = compute_calibration_plot(
        y_true=y_true_combined,
        y_pred=y_pred_prob_combined,
        from_probs=True,  # Since these are already probabilities
        save_dir=save_dir
    )
    
    # Add calibration error to summary
    cv_summary['calibration_error'] = {
        'mean': calibration_results['calibration_error'],
        'std': np.std([m['calibration_error'] for m in all_metrics])
    }
    return cv_summary
def calculate_sensitivity_specificity(targets, outputs):
    """
    Calculate sensitivity and specificity from targets and predicted outputs
    
    Args:
        targets: true labels (0/1)
        outputs: predicted probabilities
    Returns:
        dict containing sensitivity and specificity values
    """
    # Convert probability outputs to binary predictions using 0.5 threshold
    predictions = (outputs >= 0.5).astype(int)
    
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((predictions == 1) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }

def main(cfg):
    global SAVE_DIR
    SAVE_DIR = cfg.get('plot_dir') or cfg['artifact_folder']
    os.makedirs(cfg['artifact_folder'], exist_ok=True)

    # copy the config file to the artifact folder
    with open(cfg['artifact_folder'] + '/config.yaml', 'w') as f: 
        yaml.dump(cfg, f)
    
    device = torch.device(cfg['device'])
    
    # classifier network
    print("Creating video classifier network.")
    encoder = timm.create_model(**cfg['encoder_kwargs'], num_classes=0)
    if cfg.get('radimagenet_weights') is not None:
        print(f"Loading RadImageNet encoder weights from {cfg['radimagenet_weights']}")
        load_radimagenet_resnet50(encoder, cfg['radimagenet_weights'])

    # transforms
    tfms = my_transforms.VideoTransforms(cfg['res'], time_downsample_kwargs = cfg['time_downsample_kwargs'])

    # load data
    print("Preparing datasets.")
    df_frames = pd.read_csv(cfg['frame_csv'])

    # create dataset
    dat = VideoDataset(df_frames, subset_column=cfg['subset_column'], **cfg['dataset_kwargs'])
    print("Train data size:", len(dat))
    
    cv_results_test = []
    cv_results_val = []
    
    #Aggregated results store
    all_y_true = []
    all_y_pred_prob = []
    all_metrics = []
    
    for split_ix, (dl_train, dl_val, dl_test) in dat.cv_dl_gen(cfg['dataloader_kwargs'], tfms):
        
        print(f"Beginning CV split {split_ix+1} of 6:")
        
        # initialize model
        m = models.MedVidNet(copy.deepcopy(encoder), **cfg['vidnet_kwargs']).to(device)
        
        if cfg['freeze_encoder']:
            for param in m.encoder.parameters():
                param.requires_grad = False
        
        # for param in m.fc_out.parameters():
        #     param.requires_grad = False
        
        # setup optimizer
        optimizer = optim.AdamP(m.parameters(), **cfg['optim_kwargs'])
        scheduler = ReduceLROnPlateau(optimizer, **cfg['sched_kwargs'])
        stopper = EarlyStopper(**cfg['early_stopper_kwargs'])

        train_loss_ls = []
        val_loss_ls = []
        metrics_ls = []
        best_val_loss = 1e10
        best_state = None
        for epoch in range(cfg['num_epochs']):

            # train for a single epoch
            train_loss = train_one_epoch(m, optimizer, dl_train, device)
            train_loss_ls.append(train_loss)
            print(f"[{epoch} TRAIN] Cross entropy loss = {train_loss:0.5f}")       

            # evaluate
            val_loss, metrics, targets, outputs, external_ids, patient_ids = evaluate(m, dl_val, device, cv_index=split_ix)
            val_loss_ls.append(val_loss)
            
            # Calculate sensitivity and specificity for validation
            sens_spec = calculate_sensitivity_specificity(np.array(targets), np.array(outputs))
            metrics.update(sens_spec)
            
            print(f"[{epoch} VALID] Cross entropy loss = {val_loss:0.5f}")
            print(f"[{epoch} VALID] PDA classification: ", *[f"{k}={v:0.5f}" for k, v in metrics.items()])

            if val_loss < best_val_loss:
                print(f"Validation loss improved ({best_val_loss:0.5f} --> {val_loss:0.5f}). Keeping best weights in memory.")
                # best weights are held in memory only; no checkpoint files are written to disk
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                best_val_loss = val_loss

            if stopper.stop(val_loss):
                print(f"Stopping early because validation loss did not improve after {stopper.patience} epochs.")
                break
                
            scheduler.step(val_loss)
        
        print("Post-training evaluation:")
        m.load_state_dict(best_state)
        
        print("Validation:")
        val_loss, metrics, targets, outputs, external_ids, patient_ids = evaluate(m, dl_val, device, cv_index=split_ix)
        # Calculate final validation sensitivity and specificity
        sens_spec_val = calculate_sensitivity_specificity(np.array(targets), np.array(outputs))
        metrics.update(sens_spec_val)
        
        cv_results_val.append({
            'split': split_ix, 
            'loss': val_loss, 
            'metrics': metrics,
            'targets': list(targets), 
            'outputs': list(outputs),
            'external_ids': list(external_ids), 
            'patient_ids': list(patient_ids)
        })
        print(f"\tCross entropy loss = {val_loss:0.5f}")
        print("\tPDA classification: ", *[f"{k}={v:0.5f}" for k, v in metrics.items()])
        
        print("Test:")
        test_loss, metrics, targets, outputs, external_ids, patient_ids = evaluate(m, dl_test, device, cv_index=split_ix)
        # Calculate test sensitivity and specificity
        sens_spec_test = calculate_sensitivity_specificity(np.array(targets), np.array(outputs))
        metrics.update(sens_spec_test)
        
        all_y_true.append(targets)
        all_y_pred_prob.append(outputs)
        all_metrics.append(metrics)
        
        cv_results_test.append({
            'split': split_ix, 
            'loss': test_loss, 
            'metrics': metrics,
            'targets': list(targets), 
            'outputs': list(outputs),
            'external_ids': list(external_ids), 
            'patient_ids': list(patient_ids)
        })
        print(f"\tCross entropy loss = {test_loss:0.5f}")
        print("\tPDA classification: ", *[f"{k}={v:0.5f}" for k, v in metrics.items()])
    
    print("\nComputing aggregate results across all CV folds...")
    cv_summary = aggregate_cv_results(all_y_true, all_y_pred_prob, all_metrics, 
                                save_dir=cfg['artifact_folder'])
    
    # save the cross validation results
    with open(f"{cfg['artifact_folder']}/cv_results_val.json", 'w') as f:
        json.dump(cv_results_val, f, cls=CustomJSONEncoder)
        
    with open(f"{cfg['artifact_folder']}/cv_results_test.json", 'w') as f:
        json.dump(cv_results_test, f, cls=CustomJSONEncoder)
    
    with open(f"{cfg['artifact_folder']}/cv_summary.json", 'w') as f:
        json.dump(cv_summary, f, cls=CustomJSONEncoder)
        
if __name__=='__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser('Train pda video classifier')
    parser.add_argument('config', type=str, metavar='FILE',
                        help='YAML configuration file.')
    parser.add_argument('--artifact-folder', type=str, metavar='DIR', default=None,
                        help='Directory for model outputs. Overrides "artifact_folder" in config if provided.')
    parser.add_argument('--frame-csv', type=str, metavar='FILE', default=None,
                        help='CSV frame data file. Overrides "frame_csv" in config if provided.')
    parser.add_argument('--subset-column', type=str, required=False, default=None,
                        help='Name of boolean column in frames file to subset training data. Overrides "subset_column" in config if provided.')
    parser.add_argument('--num-heads', type=int, required=False, default=None,
                        help='Number of attention heads for MedVidNet model. Overrides "vidnet_kwargs: num_heads" in config if provided.')
    parser.add_argument('--pooling-method', type=str, required=False, default=None,
                        choices=['transformer', 'attn', 'tanh_attn', 'max', 'avg'],
                        help='Frame pooling method. Overrides "vidnet_kwargs: pooling_method" in config if provided.')
    parser.add_argument('--plot-dir', type=str, metavar='DIR', default=None,
                        help='Directory for metric/calibration plots. Defaults to the artifact folder.')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze the encoder (training strategy 3). Omit for full fine-tuning (strategy 1).')
    parser.add_argument('--radimagenet-weights', type=str, metavar='FILE', default=None,
                        help='RadImageNet ResNet50.pt. Overrides "radimagenet_weights" in config if provided.')
    parser.add_argument('--epochs', type=int, required=False, default=None, 
                        help="Number of training epochs.")
    parser.add_argument('--device', type=str, required=False, default=None,
                        help='Compute devie. Overrides "device" in config if provided')
    args = parser.parse_args()
    
    # load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # override with command line args
    if args.artifact_folder is not None:
        cfg['artifact_folder'] = args.artifact_folder
    if args.frame_csv is not None:
        cfg['frame_csv'] = args.frame_csv
    if args.subset_column is not None:
        cfg['subset_column'] = args.subset_column
    if args.num_heads is not None:
        cfg['vidnet_kwargs']['num_heads'] = args.num_heads
    if args.epochs is not None:
        cfg['num_epochs'] = args.epochs
    if args.pooling_method is not None:
        cfg['vidnet_kwargs']['pooling_method'] = args.pooling_method
    cfg['freeze_encoder'] = args.freeze_encoder
    if args.plot_dir is not None:
        cfg['plot_dir'] = args.plot_dir
    if args.radimagenet_weights is not None:
        cfg['radimagenet_weights'] = args.radimagenet_weights
    if args.device is not None:
        cfg['device'] = args.device
    
    print("Running training script with configuration:")
    print('-'*30)
    yaml.dump(cfg, sys.stdout)
    print('-'*30)

    main(cfg)