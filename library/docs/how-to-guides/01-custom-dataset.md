# How to Use Custom Datasets

This guide covers two ways to bring your own data: point `FolderDataset` at a folder tree, or subclass `Dataset` when your layout is different.

Every dataset yields `Sample` objects with:

- `image`: numpy array in HWC layout, `uint8`
- `masks`: numpy array of shape `(num_instances, H, W)`, `bool`, or `None`

## Folder Structure

`FolderDataset` expects images and masks in parallel trees, one subdirectory per category:

```
datasets/my-dataset/
├── images/
│   ├── backpack/
│   │   ├── 00.jpg
│   │   └── 01.jpg
│   └── barn/
│       ├── 00.jpg
│       └── 01.jpg
└── masks/
    ├── backpack/
    │   ├── 00.png
    │   └── 01.png
    └── barn/
        ├── 00.png
        └── 01.png
```

Image and mask files are paired by filename stem, so `images/backpack/00.jpg` matches `masks/backpack/00.png`. Masks are binary: any non-zero pixel belongs to the object.

Images may use `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, or `.tif`. Masks may use `.png`, `.bmp`, `.tiff`, or `.tif`.

## Using FolderDataset

```python
from pathlib import Path

from instantlearn.data.torch import FolderDataset

dataset = FolderDataset(
    root=Path("./datasets/my-dataset"),
    categories=["backpack", "barn"],  # omit to load every category
    n_shots=1,
)

sample = dataset[0]
print(sample.image.shape)      # (767, 767, 3) — HWC numpy
print(sample.masks.shape)      # (1, 767, 767) — bool
print(sample.category_labels)  # ['backpack']
```

`n_shots` controls how many images per category become references. Split the dataset and run a model:

```python
from instantlearn.models import Matcher

references = dataset.get_reference_dataset()
targets = dataset.get_target_dataset()

model = Matcher()
model.fit([references[i] for i in range(len(references))])

predictions = model.predict(targets[0])
print(predictions[0].masks.shape, predictions[0].label_names)
```

`fit()` and `predict()` accept a single `Sample`, a `list[Sample]`, or a
`Batch` — pass whichever is most convenient.

If your directory names differ, override them instead of renaming files:

```python
dataset = FolderDataset(
    root=Path("./datasets/my-dataset"),
    images_dir="Images",
    masks_dir="Annotations",
)
```

Set `masks_required=False` to include images that have no mask. This is useful for inference-only target sets.

## Creating Custom Datasets

Subclass `Dataset` when your layout does not map onto `FolderDataset`. Implement two methods:

- `_load_dataframe()` returns a `polars.DataFrame` describing the dataset
- `_load_masks(raw_sample)` returns masks as a `(num_instances, H, W)` bool array, or `None`

The dataframe needs these columns. All per-instance columns are lists, so one row can carry several objects in one image:

| Column         | Type         | Description                                        |
| -------------- | ------------ | -------------------------------------------------- |
| `image_path`   | `str`        | Path to the image file                             |
| `categories`   | `list[str]`  | Category name per instance                         |
| `category_ids` | `list[int]`  | Category id per instance                           |
| `is_reference` | `list[bool]` | Whether each instance is a reference               |
| `n_shot`       | `list[int]`  | Shot index per instance, `-1` for target instances |

Add any extra columns you need; `_load_masks` receives the whole row. The example below stores mask paths in a `mask_paths` column.

Base `__init__` does not read your data, so assign `self.df` after calling `super().__init__()`.

```python
from pathlib import Path

import numpy as np
import polars as pl

from instantlearn.data.torch import Dataset
from instantlearn.data.utils.image import read_mask


class SidecarDataset(Dataset):
    """Images in root/, each mask stored alongside as <stem>.png."""

    def __init__(self, root: Path | str, category: str, n_shots: int = 1) -> None:
        self.root = Path(root)
        self.category = category
        super().__init__(n_shots=n_shots)
        self.df = self._load_dataframe()

    def _load_dataframe(self) -> pl.DataFrame:
        rows = []
        for idx, image_path in enumerate(sorted(self.root.glob("*.jpg"))):
            mask_path = image_path.with_suffix(".png")
            if not mask_path.exists():
                continue
            is_reference = idx < self.n_shots
            rows.append({
                "image_path": str(image_path),
                "mask_paths": [str(mask_path)],
                "categories": [self.category],
                "category_ids": [0],
                "is_reference": [is_reference],
                "n_shot": [idx if is_reference else -1],
            })
        return pl.DataFrame(rows)

    def _load_masks(self, raw_sample: dict) -> np.ndarray | None:
        mask_paths = raw_sample.get("mask_paths")
        if not mask_paths:
            return None
        masks = [read_mask(path) for path in mask_paths]
        return np.stack(masks, axis=0).astype(bool)
```

Use it like any built-in dataset:

```python
dataset = SidecarDataset(root="./datasets/sidecar", category="backpack", n_shots=1)

print(len(dataset))        # 6
print(dataset.categories)  # ['backpack']

sample = dataset[0]
print(sample.image.shape)  # (767, 767, 3)
print(sample.masks.shape)  # (1, 767, 767)
```

### Common mistakes

Return masks as numpy, not torch tensors. `Sample.image` must be a numpy array in HWC layout, and conversion to model tensors happens later. Loading images with `instantlearn.data.torch.image.read_image` gives CHW torch tensors and will be rejected — use `instantlearn.data.utils.image.read_image` instead. The base class already does this for you, so you only need to care in `_load_masks`.

Cast masks to `bool`. A `uint8` mask can be scaled to 0–1 by a transform and silently become near-zero.

Keep `category_ids` consistent across rows. The same name must always map to the same id, otherwise category registration raises a `ValueError` for duplicates.
