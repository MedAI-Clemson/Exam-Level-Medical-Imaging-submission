from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import os
import pandas as pd
import numpy as np
import pydicom
from random import sample

# ----- Instance-Level and MIL Datasets -----

# --- MIL Achilles ---
class MILAchilles(Dataset):
    def __init__(self, data, data_path=None, transforms=None, save_dir=None):
        self.data = data
        self.data_path = data_path
        self.studies = data.drop_duplicates(subset=['Folder Name'])['Folder Name']
        self.labels = data.drop_duplicates(subset=['Folder Name'])['Healthy?']
        self.transform = transforms
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study_name = self.studies.iloc[idx]
        
        # read and group images with study
        img_paths = self.data['Given Name'].loc[self.data['Folder Name']==study_name]
        study_imgs = []
        for path in img_paths:
            img = read_image(self.data_path+path)
            if self.transform:
                img = self.transform(img)
            study_imgs.append(img)
        study = torch.concat(study_imgs)
        label = self.labels.iloc[idx]
        record = {
            "name": study_name,
            "study": study,
            "label": label
        }
        # return study, label
        return record
       
# --- Instance-Level Achilles ---        
class Achilles(Dataset):
    def __init__(self, data, data_path=None, transforms=None, save_dir=None):
        self.data = data
        self.data_path = data_path
        self.transform = transforms
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data['Given Name'].iloc[idx]
        image = read_image(self.data_path+img_path)
        label = self.data['Healthy?'].iloc[idx]
        patient = self.data['Folder Name'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
       
        

# --- MIL X-ray ---
class MILXray(Dataset):
    """Image CSV Dataset for Multiple Instance Learning"""
    type_map = {1: 0, 2: 1, 3: 2, 4: 2}
    inv_type_map = {0: '1-part', 1: '2-part', 2: '3- or 4-part'}

    def __init__(self, data: pd.DataFrame, data_path=None, transforms=None, seed=1, train=True, view_filter=None, save_dir=None):
        self.transforms = transforms
        self.data = data
        self.data_path = data_path
        self.train = train
        self.r = np.random.RandomState(seed=1)
        # Recode targets
        self.data['trg'] = self.data['num_parts'].map(Xray.type_map)

        # Fill nan views with "Unknown"
        self.data['view_group'] = self.data['view_group'].fillna('Unknown')
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['view_group'].isin(view_filter)]

        self.bags_list, self.labels_list, self.patient_ids_list = self._create_bags()
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

    def _create_bags(self):
        bags_list = []
        labels_list = []
        patient_ids_list = []  
        patient_ids = self.data['patient_id'].unique()  

        # Determine the maximum number of images in any bag based on the patient with the most images
        max_images = max(self.data.groupby('patient_id').size())
        #print(max_images)
        for patient_id in patient_ids:
            patient_data = self.data[self.data['patient_id'] == patient_id]
            # print(patient_data)
            bag_images = []
            bag_labels = []
            #print("$" * 40)
            # Fill the bag with images from the patient, applying transformations
            for idx, row in patient_data.iterrows():
                img = self.transforms(read_image(self.data_path+row['path']))
                trg = row['trg']
                # print(img.shape)
                bag_images.append(img)
                bag_labels.append(trg)
                #bag_labels = [1 if label == 1 else 0 for label in bag_labels]
                C, H, W = bag_images[0].shape
            bags_list.append(torch.stack(bag_images))
            bag_label = bag_labels[0]
            labels_list.append((torch.tensor(bag_label, dtype=torch.long), torch.tensor(bag_labels, dtype=torch.long)))
            patient_ids_list.append(patient_id)
        return bags_list, labels_list, patient_ids_list
    
    def __len__(self):
        #print("len")
        return len(self.bags_list)
        #return len(self.labels_list)

    def __getitem__(self, index):
        bag, (bag_label, instance_labels) = self.bags_list[index], self.labels_list[index]
        patient_id = self.patient_ids_list[index]
        #print(self.labels_list[index])
        record = {
            "name": patient_id,
            "study": bag,
            "label": bag_label
        }

        return record
    

# --- Instance-Level X-ray ---  
class Xray(Dataset):
    """Image CSV Dataset for Multiple Instance Learning"""
    type_map = {1: 0, 2: 1, 3: 2, 4: 2}
    inv_type_map = {0: '1-part', 1: '2-part', 2: '3- or 4-part'}

    def __init__(self, data: pd.DataFrame, data_path=None, transforms=None, seed=1, train=True, view_filter = None, save_dir=None):
        self.transforms = transforms
        self.data = data
        self.data_path = data_path
        self.train = train
        self.r = np.random.RandomState(seed=1)
        # Recode targets
        self.data['trg'] = self.data['num_parts'].map(Xray.type_map)

        # Fill nan views with "Unknown"
        self.data['view_group'] = self.data['view_group'].fillna('Unknown')
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['view_group'].isin(view_filter)]
    
        if save_dir is not None:
            self.data.to_csv(save_dir)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img = self.transforms(read_image(self.data_path+row['path']))
        trg = row['trg']

        return img, trg

    

# --- MIL MRI ---
class MILMRI(Dataset):
    def __init__(self, data, data_path=None, transforms=None, view_filter=None, save_dir=None, use_k=False):
        self.data = data
        self.data_path = data_path
        self.transforms = transforms
        self.use_k = use_k
        self.k_prop = 0
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['DCMSerDescr'].isin(view_filter)]
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

    def __len__(self):
        return len(self.data.drop_duplicates(subset=['ProxID'])) # by patient
        #return len(self.data) # by series

    def __getitem__(self, idx):
        #series_name = self.data['SeriesUID'].iloc[idx] # by series
        series_names = self.data.loc[self.data['ProxID'].isin(self.data.drop_duplicates(subset=['ProxID']).iloc[idx])]['SeriesUID'] # by patient
        
        # read and group images with study
        # Read the DICOM object using pydicom.
        study_imgs = []
        if self.use_k: # using the kth slice from finding label
            series_ijk = self.data.loc[self.data['ProxID'].isin(self.data.drop_duplicates(subset=['ProxID']).iloc[idx])]['ijk'].str.split()
            k_ls = [series[2] for series in series_ijk] # list of k's from all series associated with patient idx
            for i,series_name in enumerate(series_names):
                path = self.data_path+series_name
                files = [file for file in os.listdir(path) if '.dcm' in file] # getting all dicom files in a series (each file is a slice)
                temp_k = int(k_ls[i]) # temp_k is the k for the corresponding series
                k_files = [files[k] for k in range(temp_k-1,temp_k+2) if (k > 0 and k < len(files))] # Getting the k-1,k,k+1 slices
                other_files = [file for file in files if file not in k_files]
                other_files = sample(other_files, self.k_prop*len(k_files))
                for j,file in enumerate(k_files+other_files):
                    dicom = pydicom.dcmread(path+'/'+file)
                    pixels = dicom.pixel_array
                    img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
                    if self.transforms:
                        img = self.transforms(img)
                    study_imgs.append(img)
        # not using kth slice label
        else:
            for i,series_name in enumerate(series_names):
                path = self.data_path+series_name
                files = [file for file in os.listdir(path) if '.dcm' in file]
                for j,file in enumerate(files):
                    dicom = pydicom.dcmread(path+'/'+file)
                    pixels = dicom.pixel_array
                    img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
                    if self.transforms:
                        img = self.transforms(img)
                    study_imgs.append(img)
        study = torch.concat(study_imgs)
        #label = self.data['ClinSig'].iloc[idx] # by series
        #name = series_name # by series
        label = self.data.drop_duplicates(subset=['ProxID']).iloc[idx]['ClinSig']
        name = self.data.drop_duplicates(subset=['ProxID']).iloc[idx]['ProxID']
        record = {
            "name": name,
            "study": study,
            "label": label
        }
        # return study, label
        return record
    
    
# --- Instance-Level MRI ---  
class MRI(Dataset):
    def __init__(self, data, data_path=None, transforms=None, view_filter=None, save_dir=None, use_k=False):
        self.data = data
        self.data_path = data_path
        self.transforms = transforms
        self.use_k = use_k
        self.k_prop = 0
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['DCMSerDescr'].isin(view_filter)]
        
        # Creating instance level dataframe
        if self.use_k:
            temp_df = pd.DataFrame()
            num_ls = np.array([])
            for i,series in self.data.iterrows():
                path = self.data_path+series['SeriesUID']
                files = [file for file in os.listdir(path) if '.dcm' in file]
                temp_k = int(series['ijk'].split()[2])
                k_files = [files[k] for k in range(temp_k-1,temp_k+2) if (k > 0 and k < len(files))] # Getting the k-1,k,k+1 slices
                k_indices = [k for k in range(temp_k-1,temp_k+2) if (k>0 and k<len(files))]
                other_files = [file for file in files if file not in k_files] # all files except k-1,k,k+1 slices
                samples = sample(list(enumerate(other_files)), self.k_prop*len(k_files)) # getting proportion of other files other than k and k-1,k+1 slices
                sample_values = [value for index,value in samples]
                sample_indices = [index for index,value in samples]
                df = pd.DataFrame([series]*(len(k_files)+len(sample_values)), columns=self.data.columns)
                temp_df = pd.concat([temp_df, df])
                num_ls = np.append(num_ls,sample_indices+k_indices) # slide id list
            slice_id = num_ls.flatten()
            self.data = temp_df
            self.data['sliceid'] = slice_id
        else:
            temp_df = pd.DataFrame()
            num_ls = np.array([])
            count = 0
            inc = 0
            for i,series in self.data.iterrows():
                #num_slices = int(series['Dim'].split('x')[2])
                path = self.data_path+series['SeriesUID']
                files = os.listdir(path)
                files = [file for file in files if '.dcm' in file]
                num_slices = len(files)
                count+=num_slices
                df = pd.DataFrame([series]*num_slices, columns=self.data.columns)
                temp_df = pd.concat([temp_df, df])
                num_ls = np.append(num_ls,range(num_slices)) # slide id list
            slice_id = num_ls.flatten()
            self.data = temp_df
            self.data['sliceid'] = slice_id
        
        if save_dir is not None:
            self.data.to_csv(save_dir)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_name = self.data['SeriesUID'].iloc[idx]
        slice_id = int(self.data['sliceid'].iloc[idx])
        
        # read and group images with study
        # Read the DICOM object using pydicom.
        path = self.data_path+series_name
        files = os.listdir(path)
        files = [file for file in files if '.dcm' in file]
        file = files[slice_id]
        dicom = pydicom.dcmread(path+'/'+file)
        pixels = dicom.pixel_array
        img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
        if self.transforms:
            img = self.transforms(img)
        label = self.data['ClinSig'].iloc[idx]
        
        return img, label
    
    

# --- MIL CT ---  
class MILCT(Dataset):
    def __init__(self, data, data_path=None, transforms=None, view_filter=None, save_dir=None):
        self.data = data
        self.data_path = data_path
        self.transforms = transforms
        self.data.loc[self.data['Diagnosis']==2,'Diagnosis'] = 1 # changing to binary classification 2's become 1's (metastatic and normal malignant)
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_name = self.data['SeriesUID'].iloc[idx]
        
        # read and group images with study
        # Read the DICOM object using pydicom.
        path = self.data_path+series_name
        files = os.listdir(path)
        study_imgs = []
        for i,file in enumerate(files):
            if '.dcm' in file:
                dicom = pydicom.dcmread(path+'/'+file)
                pixels = dicom.pixel_array
                img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
                if self.transforms:
                    img = self.transforms(img)
                study_imgs.append(img)
        study = torch.concat(study_imgs)
        label = self.data['Diagnosis'].iloc[idx]
        record = {
            "name": series_name,
            "study": study,
            "label": label
        }
        # return study, label
        return record
    
    
# --- Instance-Level CT ---   
class CT(Dataset):
    def __init__(self, data, data_path=None, transforms=None, view_filter=None, save_dir=None):
        self.data = data
        self.data_path = data_path
        self.transforms = transforms
        self.data.loc[self.data['Diagnosis']==2,'Diagnosis'] = 1 # changing to binary classification 2's become 1's (metastatic and normal malignant)
        
        # Creating instance level dataframe
        temp_df = pd.DataFrame()
        num_ls = np.array([])
        count = 0
        for i,series in self.data.iterrows():
            path = self.data_path+series['SeriesUID']
            files = [file for file in os.listdir(path) if '.dcm' in file]
            num_slices = len(files)
            df = pd.DataFrame([series]*num_slices, columns=self.data.columns)
            temp_df = pd.concat([temp_df, df])
            num_ls = np.append(num_ls,range(num_slices))
        slice_id = num_ls.flatten()
        self.data = temp_df
        self.data['sliceid'] = slice_id
        
        if save_dir is not None:
            self.data.to_csv(save_dir)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_name = self.data['SeriesUID'].iloc[idx]
        slice_id = int(self.data['sliceid'].iloc[idx])
        
        
        # read and group images with study
        # Read the DICOM object using pydicom.
        path = self.data_path+series_name
        files = os.listdir(path)
        files = [file for file in files if '.dcm' in file]
        file = files[slice_id]
        dicom = pydicom.dcmread(path+'/'+file)
        pixels = dicom.pixel_array
        img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
        if self.transforms:
            img = self.transforms(img)
        label = self.data['Diagnosis'].iloc[idx]
        
        return img, label
    
    