# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-backed data layer for instantlearn."""

from instantlearn.data.torch.base import Dataset
from instantlearn.data.torch.coco import COCODataset
from instantlearn.data.torch.folder import FolderDataset
from instantlearn.data.torch.image import read_image, read_mask
from instantlearn.data.torch.lvis import LVISAnnotationMode, LVISDataset
from instantlearn.data.torch.per_seg import PerSegDataset
from instantlearn.data.torch.transforms import ResizeLongestSide, ToTensor

__all__ = [
    "COCODataset",
    "Dataset",
    "FolderDataset",
    "LVISAnnotationMode",
    "LVISDataset",
    "PerSegDataset",
    "ResizeLongestSide",
    "ToTensor",
    "read_image",
    "read_mask",
]
