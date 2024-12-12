"""
This script collects the Dataset classes for the CC-359, 
ACDC and M&M data sets.

Usage: serves only as a collection of individual functionalities
Authors: Rasha Sheikh, Jonathan Lennartz
"""


# - standard packages
import os
from pathlib import Path
# - third party packages
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Resize,
    CenterCrop,
    Normalize,
    functional,
)
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import preprocessing_output_dir
from nnunet.training.dataloading.dataset_loading import *

# - local source
from augment import RandAugmentWithLabels, _apply_op



class PMRIDataset(Dataset):
    """
    Multi-site dataset for prostate MRI segmentation from:
    https://liuquande.github.io/SAML/
    Possible vendors:
    - Siemens
    - GE
    - Philips
    Initialization parameters:
    - data_dir -> Path to dataset directory
    - vendor -> Vendor from possible ones to load data
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
        self._data_dir = Path(data_dir).resolve()
        self._non_empty_target = non_empty_target
        self._normalize = normalize
        self._load_data()


    def _load_data(self):
        self.input = []
        self.target = []
        vendor_sites = [
            site for site, info in self._VENDORS_INFO.items() if info[-1] == self.vendor
        ]
        for site in vendor_sites:
            site_path = self._data_dir / site
            for file in site_path.iterdir():
                # Load the ones that have a segmentation associated file to them
                if "segmentation" in file.name.lower():
                    case = file.name[4:6]
                    seg_name = "Segmentation" if site == "BMC" else "segmentation"
                    case_input_path = site_path / f"Case{case}.nii.gz"
                    case_target_path = site_path / f"Case{case}_{seg_name}.nii.gz"
                    x = nib.load(case_input_path)
                    y = nib.load(case_target_path)
                    x = torch.tensor(x.get_fdata())
                    y = torch.tensor(y.get_fdata(), dtype=torch.long)
                    self.input.append(x)
                    self.target.append(y)

        # Concatenate / Reshape to batch first / Add channel Axis
        self.input = torch.cat(self.input, dim=-1).moveaxis(-1, 0).unsqueeze(1).float()
        self.target = torch.cat(self.target, dim=-1).moveaxis(-1, 0).unsqueeze(1)
        # Relabel cases if there are two prostate classes (Since not all datasets distinguish between the two)
        self.target[self.target == 2] = 1

        if self._non_empty_target:
            non_empty_slices = self.target.sum((-1, -2, -3)) > 0
            self.input = self.input[non_empty_slices]
            self.target = self.target[non_empty_slices]


        if self._normalize:
            mean = self.input.mean()
            std = self.input.std()
            self.input = (self.input - mean) / std


    def random_split(
        self,
        val_size: float = 0.2,
    ):
        class PMRISubset(Dataset):
            def __init__(
                self,
                input,
                target,
            ):
                self.input = input
                self.target = target

            def __len__(self):
                return self.input.shape[0]
            
            def __getitem__(self, idx):
                return {
                    "input": self.input[idx], 
                    "target": self.target[idx]
                }
            
        indices = torch.randperm(len(self.input)).tolist()
        pmri_train = PMRISubset(
            input=self.input[indices[int(val_size * len(self.input)):]],
            target=self.target[indices[int(val_size * len(self.input)):]],
        )

        pmri_val = PMRISubset(
            input=self.input[indices[:int(val_size * len(self.input))]],
            target=self.target[indices[:int(val_size * len(self.input))]],
        )

        return pmri_train, pmri_val




    def __len__(self):
        return self.input.shape[0]
    

    def __getitem__(self, idx):
        return {
            "input": self.input[idx], 
            "target": self.target[idx],
            "index": idx
        }


class MNMv2Dataset(Dataset):

    def __init__(
        self,
        data_dir,
        vendor,
        binary_target: str = False,
        non_empty_target: str = True,
        normalize: str = True,
        mode="vendor",
    ):
        assert vendor in ["siemens", "ge", "philips"], "Invalid vendor"
        assert mode in ["vendor", "scanner"]
        self.vendor = vendor
        self._data_dir = Path(data_dir).resolve()
        self._binary_target = binary_target
        self._non_empty_target = non_empty_target
        self._normalize = normalize
        self._mode = mode
        if self._mode == "scanner":
            self.scanner = "SymphonyTim"

        self._data_info = pd.read_csv(
            self._data_dir / "dataset_information.csv", index_col=0
        )
        self._crop = CenterCrop(256)
        self._load_data()

    def _load_data(self):
        self.input = []
        self.target = []
        self.meta = []
        for case in self._data_info.index:
            if (
                self._mode == "vendor"
                and self.vendor in self._data_info.loc[case].VENDOR.lower()
            ) or (
                self._mode == "scanner"
                and (
                    (
                        self._train_scanner
                        and self._data_info.loc[case].SCANNER == self.scanner
                    )
                    or (
                        not self._train_scanner
                        and self._data_info.loc[case].SCANNER != self.scanner
                    )
                )
            ):
                case_path = self._data_dir / "dataset" / f"{case:03d}"
                modes = ["ES", "ED"]
                for mode in modes:
                    x = nib.load(case_path / f"{case:03d}_SA_{mode}.nii.gz")
                    y = nib.load(case_path / f"{case:03d}_SA_{mode}_gt.nii.gz")
                    x = torch.tensor(x.get_fdata()).moveaxis(-1, 0)
                    y = torch.tensor(y.get_fdata().astype(int), dtype=torch.long).moveaxis(-1, 0)
                    x = self._crop(x)
                    y = self._crop(y)
                    self.input.append(x)
                    self.target.append(y)

                if self._mode == "scanner":
                    self.meta.append(
                        [self._data_info.loc[case].SCANNER] * self.input[-1].shape[-1]
                    )

        self.input  = torch.cat(self.input,  dim=0).unsqueeze(1).float()
        self.target = torch.cat(self.target, dim=0).unsqueeze(1)

        if self._non_empty_target:
            non_empty_slices = self.target.sum((-1, -2, -3)) > 0
            self.input = self.input[non_empty_slices]
            self.target = self.target[non_empty_slices]
            if self._mode == "scanner":
                self.meta = [m for m, s in zip(self.meta, non_empty_slices) if s]

        if self._binary_target:
            self.target[self.target != 0] = 1

        if self._normalize:
            mean = self.input.mean()
            std = self.input.std()
            self.input = (self.input - mean) / std


    def random_split(
        self,
        split_ratio: float = 0.0,
        val_size: float = 0.2,
    ):
        class MNMv2Subset(Dataset):
            def __init__(
                self,
                input,
                target,
            ):
                self.input = input
                self.target = target

            def __len__(self):
                return self.input.shape[0]
            
            def __getitem__(self, idx):
                return {
                    "input": self.input[idx], 
                    "target": self.target[idx]
                }
            
        indices = torch.randperm(len(self.input)).tolist()

        if split_ratio > 0:
            train_split_idx = int((1 - val_size) * len(self.input))
            train1_split_idx = int(train_split_idx * split_ratio)

            train1 = MNMv2Subset(
                input=self.input[indices[:train1_split_idx]],
                target=self.target[indices[:train1_split_idx]],
            )

            train2 = MNMv2Subset(
                input=self.input[indices[train1_split_idx:train_split_idx]],
                target=self.target[indices[train1_split_idx:train_split_idx]],
            )

            val = MNMv2Subset(
                input=self.input[indices[train_split_idx:]],
                target=self.target[indices[train_split_idx:]],
            )

            return train1, train2, val
        else:
            mnmv2_train = MNMv2Subset(
                input=self.input[indices[int(val_size * len(self.input)):]],
                target=self.target[indices[int(val_size * len(self.input)):]],
            )

            mnmv2_val = MNMv2Subset(
                input=self.input[indices[:int(val_size * len(self.input))]],
                target=self.target[indices[:int(val_size * len(self.input))]],
            )

            return mnmv2_train, mnmv2_val


    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return {
            "input": self.input[idx], 
            "target": self.target[idx],
            "index": idx
    }
