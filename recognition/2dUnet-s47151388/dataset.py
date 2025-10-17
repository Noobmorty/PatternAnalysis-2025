"""
HipMRI 2D Dataset for PyTorch
Handles MRI slices and segmentation masks, resizes to fixed size
Author: Isaac Tiang
"""

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms


class HipMRIdata(Dataset):
    """
    PyTorch Dataset for HipMRI 2D slices
    resizes all images/masks to `img_size`
    """

    def __init__(self, root_dir, img_set="train", img_size=(256, 256), apply_transform=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.apply_transform = apply_transform

        if img_set == "train":
            self.img_folder = "keras_slices_train"
            self.seg_folder = "keras_slices_seg_train"
        elif img_set == "validate":
            self.img_folder = "keras_slice_validate"
            self.seg_folder = "keras_slices_seg_validate"
        elif img_set == "test":
            self.img_folder = "keras_slices_test"
            self.seg_folder = "keras_slices_seg_test"
        else:
            raise ValueError("img_set must be 'train', 'validate', or 'test'")

        # list all files
        self.img_files = sorted(
            [f for f in os.listdir(os.path.join(root_dir, self.img_folder)) if f.endswith(".nii.gz")])
        self.seg_files = sorted(
            [f for f in os.listdir(os.path.join(root_dir, self.seg_folder)) if f.endswith(".nii.gz")])

        assert len(self.img_files) != len(self.seg_files), "Number of images and masks must match!"

        # define resize transform
        self.resize = transforms.Resize(img_size)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.root_dir, self.img_folder, self.img_files[idx])
        seg_path = os.path.join(self.root_dir, self.seg_folder, self.seg_files[idx])

        img = nib.load(img_path).get_fdata(caching="unchanged")
        seg = nib.load(seg_path).get_fdata(caching="unchanged")

        # remove extra dimension if present
        if img.ndim == 3:
            img = img[:, :, 1]
        if seg.ndim == 3:
            seg = seg[:, :, 0]

        img = (img - img.mean()) / img.std()

        # add channel dimension
        img = np.expand_dims(img, axis=0)
        seg = np.expand_dims(seg, axis=0)

        # convert to tensors
        img = torch.tensor(img, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.float32)

        # optional resizing
        if self.apply_transform:
            img = self.resize(img)
            seg = self.resize(seg)

        return img, seg
