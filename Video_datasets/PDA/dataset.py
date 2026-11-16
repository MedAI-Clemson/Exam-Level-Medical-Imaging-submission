import torch
from torch.utils.data import Dataset, default_collate
from torch.utils.data import DataLoader
from torchvision.io import read_image
import pandas as pd
import random
import copy
from torch.nn.utils.rnn import pad_sequence

class PdaVideos(Dataset):
    """PDA Video dataset used to train video-based model"""
    
    # define mapping dictionaries
    type_map = {'nopda': 0, 'pda': 1}
    view_map = {'nonPDAView': 0, 'pdaRelatedView': 1, 'pdaView': 2}
    mode_map = {'2d': 0, 'color': 1, 'color_compare': 2}
    
    # define inverse maps
    inv_type_map = {v: k for k, v in type_map.items()}
    inv_view_map = {v: k for k, v in view_map.items()}
    inv_mode_map = {v: k for k, v in mode_map.items()}

    def __init__(self, 
                 data: pd.DataFrame, 
                 transforms = None, 
                 split = None,
                 view_filter=None, 
                 mode_filter=None, 
                 subset_column=None):
        
        self.data = data
        
        # subset to split
        if split is not None:
            self.data = data.query('Split==@split').copy() 
        
        # optionally subset by subset column
        if subset_column is not None:
            self.data = self.data[self.data[subset_column]].copy()

        # recode targets
        self.data.loc[:,'trg_type'] = self.data['patient_type'].map(self.type_map)
        self.data.loc[:,'trg_view'] = self.data['view'].map(self.view_map)
        self.data.loc[:,'trg_mode'] = self.data['mode'].map(self.mode_map)

        # set transforms
        if transforms is not None:
            self.tfms = transforms

        # apply filters
        if view_filter is not None:
            self.data = self.data.loc[self.data['view'].isin(view_filter)]
        if mode_filter is not None:
            self.data = self.data.loc[self.data['mode'].isin(mode_filter)]
            
        self.data = self.data.reset_index()

        # Study-level data (for length and indexing)
        self.study_data = self.data[[
            'study', 'patient_id', 'patient_type', 'trg_type', 'cv_split'
            ]].drop_duplicates().reset_index()

        # Keep video_data for getting complete video information
        self.video_data = self.data[[
            'study', 'patient_id', 'patient_type', 'external_id', 
            'cv_split', 'trg_type', 'trg_view', 'trg_mode'
            ]].drop_duplicates().reset_index()

    def __getitem__(self, index) -> dict:
        study_row = self.study_data.iloc[index]
        
        # Get all videos for this study
        study_videos = self.video_data[self.video_data['study'] == study_row['study']]
        
        videos = []
        video_metadata = []
        
        # Process each video in study (keeping original video formation logic)
        for _, video_row in study_videos.iterrows():
            # Original frame-to-video logic
            is_frame_from_vid = (self.data.patient_type == video_row['patient_type']) & \
                              (self.data.external_id == video_row['external_id'])
            frames = self.data.loc[is_frame_from_vid, 'png_path'].tolist()
            frames = sorted(frames)
            
            # Create video
            video = torch.concat([read_image(f)[None] for f in frames])
            if hasattr(self, 'tfms'):
                video = self.tfms(video)
            videos.append(video)
            
            # Store video metadata
            video_metadata.append({
                'external_id': video_row['external_id'],
                'trg_view': int(video_row['trg_view']),
                'trg_mode': int(video_row['trg_mode'])
            })

        # Build the study-level record
        record = {
            'videos': videos,  # List of video tensors
            'video_metadata': video_metadata,  # List of metadata dicts for each video
            'study': study_row['study'],
            'patient': study_row['patient_id'],
            'trg_type': int(study_row['trg_type'])  # Study-level label
        }
        
        return record

    def __len__(self):
        return len(self.study_data)
    
    def cv_dl_gen(self, dataloader_kwargs, tfms):
        for split_ix in range(6):
            # randomly choose a validation index
            val_ix = random.choice([j for j in range(6) if j != split_ix])
            
            # identify which rows are in which split
            test_cond = (self.study_data.cv_split==split_ix)
            val_cond = (self.study_data.cv_split==val_ix)
            train_cond = ~test_cond & ~val_cond
            
            # get the indices of the samples in each split
            train_ix = self.study_data[train_cond].index
            val_ix = self.study_data[val_cond].index
            test_ix = self.study_data[test_cond].index
            
            # get the datasets for each
            d_train = torch.utils.data.Subset(self, train_ix)
            d_val = torch.utils.data.Subset(self, val_ix)
            d_test = torch.utils.data.Subset(self, test_ix)
            
            # set the transforms appropriately
            d_train.dataset = copy.copy(self)
            
            # get the split-specific transforms and set them
            tfms_train = tfms.get_transforms('train')
            tfms_test = tfms.get_transforms('test')
            d_train.dataset.tfms = tfms_train
            d_val.dataset.tfms = tfms_test
            d_test.dataset.tfms = tfms_test
            
            # make the dataloaders
            dl_train = DataLoader(d_train, shuffle=True, collate_fn=collate_study, **dataloader_kwargs)
            dl_val = DataLoader(d_val, collate_fn=collate_study, **dataloader_kwargs)
            dl_test = DataLoader(d_test, collate_fn=collate_study, **dataloader_kwargs)
            
            yield split_ix, (dl_train, dl_val, dl_test)

            
def collate_study(batch_list):
    """
    Custom collate function that:
    - Concatenates all videos within each study
    - Tracks video boundaries with markers 
    - Returns tensor in [sum(videos_per_study), C, H, W] format
    """
    study_frames = []  # Store concatenated frames for each study
    videos_per_study = []  # Track videos per study
    all_metadata = []
    video_markers = []  # Track start/end of each video within studies
    
    for batch in batch_list:
        # Process videos in this study
        study_videos = batch['videos']
        videos_per_study.append(len(study_videos))
        
        # Track markers for videos in this study
        study_markers = []
        curr_pos = 0
        for video in study_videos:
            t = video.shape[0]  # Length of current video
            study_markers.extend([curr_pos, curr_pos + t])  # [start, end] for each video
            curr_pos += t
            
        video_markers.append(study_markers)
        
        # Concatenate all frames in this study
        study_frames_tensor = torch.cat(study_videos, dim=0)  # [total_T, C, H, W] for this study
        study_frames.append(study_frames_tensor)
        
        all_metadata.extend(batch['video_metadata'])
    
    # Concatenate all studies into one tensor
    videos_tensor = torch.cat(study_frames, dim=0)  # [sum(total_T), C, H, W]
    
    record = {
        'videos': videos_tensor,                   # [sum(total_T), C, H, W]
        'video_markers': video_markers,            # List of [start,end] markers for each video in each study
        'videos_per_study': videos_per_study,      # Number of videos per study
        'video_metadata': all_metadata             # Metadata for all videos
    }
    
    # Add study-level information
    study_info = [{k: v for k, v in b.items() 
                   if k not in ['videos', 'video_metadata']} 
                  for b in batch_list]
    record.update(default_collate(study_info))
    
    return record