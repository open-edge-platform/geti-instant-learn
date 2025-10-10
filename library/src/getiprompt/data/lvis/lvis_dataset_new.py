# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LVIS dataset."""

from logging import getLogger
from pathlib import Path

import lvis

import torch
from enum import Enum

logger = getLogger("Geti Prompt")


class Subset(Enum):
    """Enum for subset of the dataset."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RefactorLVISDataset(torch.utils.data.Dataset):
    """This class represents the LVIS dataset.

    Args:
        root_path: The path to the root directory of the LVIS dataset.
        whitelist: The classes that are selected, if empty, all categories are used
        download_full_dataset: If True download the full dataset otherwise,
                               each image is downloaded on demand.
        copy_files: If the full dataset is download then copy_files will copy files from the
                    COCO dataset to the LVIS dataset folders.
                    If copy_files is True, then after copying, download_full_dataset can be set to false.
        iterator_kwargs: Keyword arguments passed to the iterator_type
    """

    def __init__(
        self,
        root_path: str | Path = "~/data/lvis",
        whitelist: str | list[str] = ("cupcake", "sheep", "pastry", "doughnut"),
        name: Subset = Subset.TRAIN,
        num_samples: int = -1,
    ) -> None:
        super().__init__()
        self._root_path = Path(root_path).expanduser()
        Path(self._root_path).mkdir(parents=True, exist_ok=True)

        subset_info = {
            Subset.TRAIN: {
                "anno_files": Path(self._root_path) / "lvis_v1_train.json",
                "sources": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
                "downloads": "http://images.cocodataset.org/zips/train2017.zip",
                "source_folders": Path(self._root_path) / "downloads" / "train2017",
            },
            Subset.VAL: {
                "anno_files": Path(self._root_path) / "lvis_v1_val.json",
                "sources": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
                "downloads": "http://images.cocodataset.org/zips/val2017.zip",
                "source_folders": Path(self._root_path) / "downloads" / "val2017",
            },
        }

        target_labels = whitelist

        self.img_infos = self.load_data(subset_info[name], target_labels, num_samples)
        self.subset = name
    
    def load_data(
        self, 
        subset: dict[str, str | Path],
        target_labels: list[str],
        num_samples: int = -1,
    ) -> None:
        dataset = lvis.LVIS(subset["anno_files"])
        cats = dataset.load_cats(dataset.get_cat_ids())
        label_names = [c['name'] for c in cats]
        label_ids = [c['id'] for c in cats]
        tgt_label_ids = []
        
        for target_label in target_labels:
            if target_label not in label_names:
                msg = f"Target label {target_label} not found in the dataset"
                raise ValueError(msg)
            target_label_id = label_ids[label_names.index(target_label)]
            tgt_label_ids.append(target_label_id)

        cats = dataset.cats
        
        img_ids = []
        for cid in tgt_label_ids:
            img_ids.extend(dataset.cat_img_map[cid])
        img_ids = list(set(img_ids))
        if num_samples > 0:
            img_ids = img_ids[:num_samples]
        img_infos = dataset.load_imgs(img_ids)
        ann_ids = dataset.get_ann_ids(img_ids=img_ids, cat_ids=tgt_label_ids)
        return img_infos

    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, index):
        return self.img_infos[index]



if __name__ == "__main__":
    dataset = RefactorLVISDataset(
        root_path="~/data/lvis",
        whitelist=["cupcake", "sheep", "pastry", "doughnut"],
        name=Subset.TRAIN,
        num_samples=10,
    )
