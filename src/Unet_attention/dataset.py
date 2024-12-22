# """
# This script collects the Dataset classes for the CC-359, 
# ACDC and M&M data sets.

# Usage: serves only as a collection of individual functionalities
# Authors: Rasha Sheikh, Jonathan Lennartz
# """


# # - standard packages
# import os
# from pathlib import Path
# # - third party packages
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision.transforms import (
#     Resize,
#     CenterCrop,
#     Normalize,
#     functional,
# )
# from sklearn.preprocessing import MinMaxScaler
# import nibabel as nib
# from batchgenerators.utilities.file_and_folder_operations import *
# from nnunet.paths import preprocessing_output_dir
# from nnunet.training.dataloading.dataset_loading import *

# # - local source
# from augment import RandAugmentWithLabels, _apply_op



# class PMRIDataset(Dataset):
#     """
#     Multi-site dataset for prostate MRI segmentation from:
#     https://liuquande.github.io/SAML/
#     Possible vendors:
#     - Siemens
#     - GE
#     - Philips
#     Initialization parameters:
#     - data_dir -> Path to dataset directory
#     - vendor -> Vendor from possible ones to load data
#     """

#     _VENDORS_INFO = {
#         "RUNMC": ["A", "siemens"],
#         "BMC": ["B", "philips"],
#         "I2CVB": ["C", "siemens"],
#         "UCL": ["D", "siemens"],
#         "BIDMC": ["E", "ge"],
#         "HK": ["F", "siemens"],
#     }
#     _DS_CONFIG = {"num_classes": 2, "spatial_dims": 2, "size": (384, 384)}

#     def __init__(
#         self,
#         data_dir: str,
#         vendor: str,
#         non_empty_target: bool = True,
#         normalize: bool = True,
#     ):
#         assert vendor in ["siemens", "ge", "philips"], "Invalid vendor"
#         self.vendor = vendor
#         self._data_dir = Path(data_dir).resolve()
#         self._non_empty_target = non_empty_target
#         self._normalize = normalize
#         self._load_data()


#     def _load_data(self):
#         self.input = []
#         self.target = []
#         vendor_sites = [
#             site for site, info in self._VENDORS_INFO.items() if info[-1] == self.vendor
#         ]
#         for site in vendor_sites:
#             site_path = self._data_dir / site
#             for file in site_path.iterdir():
#                 # Load the ones that have a segmentation associated file to them
#                 if "segmentation" in file.name.lower():
#                     case = file.name[4:6]
#                     seg_name = "Segmentation" if site == "BMC" else "segmentation"
#                     case_input_path = site_path / f"Case{case}.nii.gz"
#                     case_target_path = site_path / f"Case{case}_{seg_name}.nii.gz"
#                     x = nib.load(case_input_path)
#                     y = nib.load(case_target_path)
#                     x = torch.tensor(x.get_fdata())
#                     y = torch.tensor(y.get_fdata(), dtype=torch.long)
#                     self.input.append(x)
#                     self.target.append(y)

#         # Concatenate / Reshape to batch first / Add channel Axis
#         self.input = torch.cat(self.input, dim=-1).moveaxis(-1, 0).unsqueeze(1).float()
#         self.target = torch.cat(self.target, dim=-1).moveaxis(-1, 0).unsqueeze(1)
#         # Relabel cases if there are two prostate classes (Since not all datasets distinguish between the two)
#         self.target[self.target == 2] = 1

#         if self._non_empty_target:
#             non_empty_slices = self.target.sum((-1, -2, -3)) > 0
#             self.input = self.input[non_empty_slices]
#             self.target = self.target[non_empty_slices]


#         if self._normalize:
#             mean = self.input.mean()
#             std = self.input.std()
#             self.input = (self.input - mean) / std


#     def random_split(
#         self,
#         val_size: float = 0.2,
#     ):
#         class PMRISubset(Dataset):
#             def __init__(
#                 self,
#                 input,
#                 target,
#             ):
#                 self.input = input
#                 self.target = target

#             def __len__(self):
#                 return self.input.shape[0]
            
#             def __getitem__(self, idx):
#                 return {
#                     "input": self.input[idx], 
#                     "target": self.target[idx]
#                 }
            
#         indices = torch.randperm(len(self.input)).tolist()
#         pmri_train = PMRISubset(
#             input=self.input[indices[int(val_size * len(self.input)):]],
#             target=self.target[indices[int(val_size * len(self.input)):]],
#         )

#         pmri_val = PMRISubset(
#             input=self.input[indices[:int(val_size * len(self.input))]],
#             target=self.target[indices[:int(val_size * len(self.input))]],
#         )

#         return pmri_train, pmri_val




#     def __len__(self):
#         return self.input.shape[0]
    

#     def __getitem__(self, idx):
#         return {
#             "input": self.input[idx], 
#             "target": self.target[idx],
#             "index": idx
#         }


# class MNMv2Dataset(Dataset):

#     def __init__(
#         self,
#         data_dir,
#         vendor,
#         binary_target: str = False,
#         non_empty_target: str = True,
#         normalize: str = True,
#         mode="vendor",
#     ):
#         assert vendor in ["siemens", "ge", "philips"], "Invalid vendor"
#         assert mode in ["vendor", "scanner"]
#         self.vendor = vendor
#         self._data_dir = Path(data_dir).resolve()
#         self._binary_target = binary_target
#         self._non_empty_target = non_empty_target
#         self._normalize = normalize
#         self._mode = mode
#         if self._mode == "scanner":
#             self.scanner = "SymphonyTim"

#         self._data_info = pd.read_csv(
#             self._data_dir / "dataset_information.csv", index_col=0
#         )
#         self._crop = CenterCrop(256)
#         self._load_data()

#     def _load_data(self):
#         self.input = []
#         self.target = []
#         self.meta = []
#         for case in self._data_info.index:
#             if (
#                 self._mode == "vendor"
#                 and self.vendor in self._data_info.loc[case].VENDOR.lower()
#             ) or (
#                 self._mode == "scanner"
#                 and (
#                     (
#                         self._train_scanner
#                         and self._data_info.loc[case].SCANNER == self.scanner
#                     )
#                     or (
#                         not self._train_scanner
#                         and self._data_info.loc[case].SCANNER != self.scanner
#                     )
#                 )
#             ):
#                 case_path = self._data_dir / "dataset" / f"{case:03d}"
#                 modes = ["ES", "ED"]
#                 for mode in modes:
#                     x = nib.load(case_path / f"{case:03d}_SA_{mode}.nii.gz")
#                     y = nib.load(case_path / f"{case:03d}_SA_{mode}_gt.nii.gz")
#                     x = torch.tensor(x.get_fdata()).moveaxis(-1, 0)
#                     y = torch.tensor(y.get_fdata().astype(int), dtype=torch.long).moveaxis(-1, 0)
#                     x = self._crop(x)
#                     y = self._crop(y)
#                     self.input.append(x)
#                     self.target.append(y)

#                 if self._mode == "scanner":
#                     self.meta.append(
#                         [self._data_info.loc[case].SCANNER] * self.input[-1].shape[-1]
#                     )

#         self.input  = torch.cat(self.input,  dim=0).unsqueeze(1).float()
#         self.target = torch.cat(self.target, dim=0).unsqueeze(1)

#         if self._non_empty_target:
#             non_empty_slices = self.target.sum((-1, -2, -3)) > 0
#             self.input = self.input[non_empty_slices]
#             self.target = self.target[non_empty_slices]
#             if self._mode == "scanner":
#                 self.meta = [m for m, s in zip(self.meta, non_empty_slices) if s]

#         if self._binary_target:
#             self.target[self.target != 0] = 1

#         if self._normalize:
#             mean = self.input.mean()
#             std = self.input.std()
#             self.input = (self.input - mean) / std


#     def random_split(
#         self,
#         val_size: float = 0.2,
#     ):
#         class MNMv2Subset(Dataset):
#             def __init__(
#                 self,
#                 input,
#                 target,
#             ):
#                 self.input = input
#                 self.target = target

#             def __len__(self):
#                 return self.input.shape[0]
            
#             def __getitem__(self, idx):
#                 return {
#                     "input": self.input[idx], 
#                     "target": self.target[idx]
#                 }
            
#         indices = torch.randperm(len(self.input)).tolist()
#         mnmv2_train = MNMv2Subset(
#             input=self.input[indices[int(val_size * len(self.input)):]],
#             target=self.target[indices[int(val_size * len(self.input)):]],
#         )

#         mnmv2_val = MNMv2Subset(
#             input=self.input[indices[:int(val_size * len(self.input))]],
#             target=self.target[indices[:int(val_size * len(self.input))]],
#         )

#         return mnmv2_train, mnmv2_val


#     def __len__(self):
#         return self.input.shape[0]

#     def __getitem__(self, idx):
#         return {
#             "input": self.input[idx], 
#             "target": self.target[idx],
#             "index": idx
#     }



"""
This script contains dataset classes for prostate MRI segmentation (PMRI) 
and MNMv2 multi-vendor segmentation tasks. These datasets support:
- Data normalization
- Slicing based on target presence
- Splitting into training and validation subsets
- Integration with self-supervised learning workflows

Authors: Rasha Sheikh, Jonathan Lennartz
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Normalize, functional as F
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib


class PMRIDataset(Dataset):
    """
    Multi-site dataset for prostate MRI segmentation.
    Supports loading data from different vendors (Siemens, GE, Philips).
    """

    _VENDORS_INFO = {
        "RUNMC": ["A", "siemens"],
        "BMC": ["B", "philips"],
        "I2CVB": ["C", "siemens"],
        "UCL": ["D", "siemens"],
        "BIDMC": ["E", "ge"],
        "HK": ["F", "siemens"],
    }
    _DS_CONFIG = {"num_classes": 2, "spatial_dims": 2, "size": (384, 384)}

    def __init__(
        self,
        data_dir: str,
        vendor: str,
        non_empty_target: bool = True,
        normalize: bool = True,
    ):
        assert vendor in ["siemens", "ge", "philips"], "Invalid vendor"
        self.vendor = vendor
        self.data_dir = Path(data_dir).resolve()
        self.non_empty_target = non_empty_target
        self.normalize = normalize
        self.input = None
        self.target = None
        self._load_data()

    def _load_data(self):
        inputs, targets = [], []
        vendor_sites = [
            site for site, info in self._VENDORS_INFO.items() if info[-1] == self.vendor
        ]

        for site in vendor_sites:
            site_path = self.data_dir / site
            for file in site_path.iterdir():
                if "segmentation" in file.name.lower():
                    case = file.name[4:6]
                    seg_name = "Segmentation" if site == "BMC" else "segmentation"
                    input_path = site_path / f"Case{case}.nii.gz"
                    target_path = site_path / f"Case{case}_{seg_name}.nii.gz"
                    
                    # Load data using nibabel
                    x = nib.load(input_path).get_fdata()
                    y = nib.load(target_path).get_fdata()
                    
                    # Convert to tensors
                    x = torch.tensor(x).unsqueeze(0)  # Add channel axis
                    y = torch.tensor(y, dtype=torch.long).unsqueeze(0)

                    inputs.append(x)
                    targets.append(y)

        # Concatenate and apply preprocessing
        self.input = torch.cat(inputs).float()
        self.target = torch.cat(targets)

        # Relabel targets if necessary
        self.target[self.target == 2] = 1

        # Filter non-empty targets
        if self.non_empty_target:
            non_empty_slices = self.target.sum(dim=(1, 2, 3)) > 0
            self.input = self.input[non_empty_slices]
            self.target = self.target[non_empty_slices]

        # Normalize inputs
        if self.normalize:
            self.input = (self.input - self.input.mean()) / self.input.std()

    def random_split(self, val_size: float = 0.2):
        indices = torch.randperm(len(self.input))
        split_idx = int(val_size * len(self.input))
        
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]
        return (
            self._subset(train_indices),
            self._subset(val_indices),
        )

    def _subset(self, indices):
        return PMRISubset(self.input[indices], self.target[indices])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "index": idx,
        }


class PMRISubset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
        }


class MNMv2Dataset(Dataset):
    """
    Dataset for multi-vendor segmentation tasks with options for binary targets,
    normalization, and training/validation splits.
    """

    def __init__(
        self,
        data_dir: str,
        vendor: str,
        binary_target: bool = False,
        non_empty_target: bool = True,
        normalize: bool = True,
        mode="vendor",
    ):
        assert vendor in ["siemens", "ge", "philips"], "Invalid vendor"
        assert mode in ["vendor", "scanner"], "Invalid mode"
        self.vendor = vendor
        self.data_dir = Path(data_dir).resolve()
        self.binary_target = binary_target
        self.non_empty_target = non_empty_target
        self.normalize = normalize
        self.mode = mode

        self.data_info = pd.read_csv(self.data_dir / "dataset_information.csv", index_col=0)
        self.crop = CenterCrop(256)
        self.input = None
        self.target = None
        self._load_data()

    def _load_data(self):
        inputs, targets = [], []

        for case in self.data_info.index:
            case_vendor = self.data_info.loc[case, "VENDOR"].lower()
            if self.mode == "vendor" and self.vendor in case_vendor:
                case_path = self.data_dir / "dataset" / f"{case:03d}"
                for mode in ["ES", "ED"]:
                    x = nib.load(case_path / f"{case:03d}_SA_{mode}.nii.gz").get_fdata()
                    y = nib.load(case_path / f"{case:03d}_SA_{mode}_gt.nii.gz").get_fdata()

                    # Convert to tensors
                    x = torch.tensor(x).moveaxis(-1, 0)  # Channels first
                    y = torch.tensor(y.astype(int), dtype=torch.long).moveaxis(-1, 0)

                    # Apply cropping
                    x = self.crop(x)
                    y = self.crop(y)

                    inputs.append(x)
                    targets.append(y)

        # Concatenate and preprocess
        self.input = torch.cat(inputs).unsqueeze(1).float()
        self.target = torch.cat(targets).unsqueeze(1)

        # Filter non-empty targets
        if self.non_empty_target:
            non_empty_slices = self.target.sum(dim=(1, 2, 3)) > 0
            self.input = self.input[non_empty_slices]
            self.target = self.target[non_empty_slices]

        # Apply binary targets if needed
        if self.binary_target:
            self.target[self.target != 0] = 1

        # Normalize inputs
        if self.normalize:
            self.input = (self.input - self.input.mean()) / self.input.std()

    def random_split(self, val_size: float = 0.2):
        indices = torch.randperm(len(self.input))
        split_idx = int(val_size * len(self.input))
        
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]
        return (
            self._subset(train_indices),
            self._subset(val_indices),
        )

    def _subset(self, indices):
        return MNMv2Subset(self.input[indices], self.target[indices])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "index": idx,
        }


class MNMv2Subset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
        }
