import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.io import read_video
import pandas as pd
import random
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.io import read_video
import pandas as pd
import random
import copy
import numpy as np

class VideoDataset(Dataset):
    """Video dataset for quality assessment"""
    
    quality_map = {0: 'bad', 1: 'good'}
    inv_quality_map = {'bad': 0, 'good': 1}
    
    def __init__(self, 
                 data: pd.DataFrame,
                 transforms=None,
                 split=None,
                 subset_column=None,
                 resolution_filter=None):
        
        self.data = data
        
        # subset to split
        if split is not None:
            self.data = data.query('Split==@split').copy()
            
        # optionally subset by subset column
        if subset_column is not None:
            self.data = self.data[self.data[subset_column]].copy()
            
        # Apply resolution filter if provided    
        if resolution_filter is not None:
            width, height = resolution_filter
            self.data = self.data[
                (self.data['Width'] == width) & 
                (self.data['Height'] == height)
            ].copy()
            
        # Add quality labels    
        self.data.loc[:,'quality_label'] = (self.data['Quality'] >= 3).astype(int)
        
        # Set transforms
        if transforms is not None:
            self.tfms = transforms
            
        self.data = self.data.reset_index()
        
        # Group to get video-level data
        self.video_data = self.data[[
            'NewPatientID', 'VideoName', 'VideoPath',
            'Quality', 'quality_label'
        ]].drop_duplicates().reset_index()
        
        
    def __getitem__(self, idx):
        row = self.video_data.iloc[idx]

        # Read video and keep original format
        video, _, _ = read_video(row['VideoPath'], pts_unit='sec')  # Shape: [T, H, W, C]

        # Permute once to get [T, C, H, W] format
        video = video.permute(0, 3, 1, 2)  # Shape: [T, C, H, W]

        # Apply transforms if they exist
        if hasattr(self, 'tfms'):
            video = self.tfms(video)

        # Build output dictionary
        record = dict()
        record['video'] = video  # This will now be [T, C, H, W]
        record['patient_id'] = row['NewPatientID']
        record['video_name'] = row['VideoName']
        record['quality'] = row['Quality']
        record['quality_label'] = int(row['quality_label'])
        record['num_frames'] = video.shape[0]

        return record

    
    def __len__(self):
        return len(self.video_data)
    
    def cv_dl_gen(self, dataloader_kwargs, tfms):
        video_names = self.video_data['VideoName'].unique()
        splits = {video: i % 6 for i, video in enumerate(video_names)}
        self.video_data['cv_split'] = self.video_data['VideoName'].map(splits)
        
        for split_ix in range(6):
            val_ix = random.choice([j for j in range(6) if j != split_ix])
            
            test_cond = (self.video_data['cv_split']==split_ix)
            val_cond = (self.video_data['cv_split']==val_ix)
            train_cond = ~test_cond & ~val_cond
            
            train_ix = self.video_data[train_cond].index
            val_ix = self.video_data[val_cond].index
            test_ix = self.video_data[test_cond].index
            
            d_train = torch.utils.data.Subset(self, train_ix)
            d_val = torch.utils.data.Subset(self, val_ix)
            d_test = torch.utils.data.Subset(self, test_ix)
            
            d_train.dataset = copy.copy(self)
            
            tfms_train = tfms.get_transforms('train')
            tfms_test = tfms.get_transforms('test')
            d_train.dataset.tfms = tfms_train
            d_val.dataset.tfms = tfms_test
            d_test.dataset.tfms = tfms_test
            
            dl_train = DataLoader(d_train, shuffle=True, collate_fn=collate_video_batch, drop_last=True, **dataloader_kwargs)
            dl_val = DataLoader(d_val, collate_fn=collate_video_batch, drop_last=True,**dataloader_kwargs)
            dl_test = DataLoader(d_test, collate_fn=collate_video_batch, drop_last=True, **dataloader_kwargs)
            
            yield split_ix, (dl_train, dl_val, dl_test)
            
def collate_video_batch(batch_list):
    """Custom collate function for variable length videos"""
    videos = [b['video'] for b in batch_list]
    videos = torch.cat(videos, dim=0)  
    
    patient_ids = [b['patient_id'] for b in batch_list]
    unique_patients = list(set(patient_ids))
    patient_to_idx = {pid: idx for idx, pid in enumerate(unique_patients)}
    
    record = {
        'video': videos,
        'video_lengths': [b['video'].shape[0] for b in batch_list],
        # Map each video to its study index - videos from same study get same index
        'study_indices': torch.tensor([patient_to_idx[pid] for pid in patient_ids])
    }
    
    # Handle other data
    for b in batch_list:
        b.pop('video')
    record.update(default_collate(batch_list))
    
    return record