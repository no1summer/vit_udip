"""
Data handling for ViT-UDIP.

This module provides dataset classes and data loading utilities
for medical image processing.
"""

import torch
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset


class MedicalImageDataset(Dataset):
    """Dataset for medical images with padding and normalization.
    
    Args:
        datafile (str): Path to CSV file containing image paths
        modality (str): Column name containing image paths
    """
    
    def __init__(self, datafile, modality):
        """
        Args:
            datafile (str): Path to CSV file or list of file paths
            modality (str): Column containing location of modality of interest
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
        """
        self.datafile = pd.read_csv(datafile)
        self.unbiased_brain = self.datafile[modality]

    def __len__(self):
        return len(self.unbiased_brain)

    def __getitem__(self, idx):
        """Get item from dataset.
        
        Args:
            idx (int): Index of item to retrieve
            
        Returns:
            tuple: (image_tensor, mask_tensor)
        """
        img_name = self.unbiased_brain[idx]
        img = nib.load(img_name)
        img = img.get_fdata()
        img = torch.from_numpy(img)
        img = torch.nn.functional.pad(img, (0,0,3,3,0,0))  # padding image from 182x218x182 to 182x224x182
        # padding needs to be done before normalization
        mask = img != 0
        img = (img - img[img != 0].mean()) / img[img != 0].std()
        img = img.type(torch.float)
        return img, mask
