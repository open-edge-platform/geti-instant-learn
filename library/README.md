# 👋 Introduction

The Geti Prompt Library provides a robust platform for experimenting with visual prompting techniques. Its modular pipeline design allows researchers and developers to easily combine, swap, and extend components such as backbone networks, feature extractors, matching algorithms, and mask generators.

## 📦 Installation

```bash
cd library

# With CUDA support (recommended for GPU)
uv sync --extra gpu

# CPU only
uv sync --extra cpu

# Intel XPU
uv sync --extra xpu
```

<details>
<summary><strong>💡 Advanced: Install with extras</strong></summary>

```bash
# Install with xFormers for faster inference
uv sync --extra extras

# Install development dependencies
uv sync --extra dev

# Install all dependencies
uv sync --extra full
```

</details>

## 🚀 Quick Start

### Python API

<p align="center">
  <img src="tests/assets/fss-1000/images/apple/1.jpg" width="200" alt="Reference">
  <img src="tests/assets/fss-1000/masks/apple/1.png" width="200" alt="Mask">
  <img src="tests/assets/fss-1000/images/apple/2.jpg" width="200" alt="Target">
</p>
<p align="center"><i>Reference → Mask → Target</i></p>

**Generate a reference mask with SAM:**

```python
import torch
from getiprompt.components.sam import PyTorchSAMPredictor
from getiprompt.utils.constants import SAMModelName
from getiprompt.data.utils import read_image

# Load reference image
ref_image = read_image("library/tests/assets/fss-1000/images/apple/1.jpg")

# Initialize SAM predictor (auto-downloads weights)
predictor = PyTorchSAMPredictor(SAMModelName.SAM_HQ_TINY, device="cuda")

# Set image and generate mask from a point click
predictor.set_image(ref_image)
ref_mask, _, _ = predictor.predict(
    point_coords=torch.tensor([[[51, 150]]], device="cuda"),  # Click on apple
    point_labels=torch.tensor([[1]], device="cuda"),           # 1 = foreground
    multimask_output=False,
)
```

**Fit and predict with Matcher:**

```python
from getiprompt.models import Matcher
from getiprompt.data import Batch, Sample
from getiprompt.data.utils import read_image

# Initialize Matcher
model = Matcher(device="cuda")

# Create reference sample with the generated mask
ref_sample = Sample(
    image=ref_image,
    masks=ref_mask[0],
    categories=["apple"],
    category_ids=[0],
)

# Fit on reference
model.fit(Batch.collate([ref_sample]))

# Predict on target image
target_image = read_image("library/tests/assets/fss-1000/images/apple/2.jpg")
target_sample = Sample(image=target_image)
predictions = model.predict(Batch.collate([target_sample]))

# Access results
masks = predictions[0]["pred_masks"]   # Predicted segmentation masks
```

**Fit and predict with GroundedSAM:**

```python
from getiprompt.models import GroundedSAM
from getiprompt.data import Batch, Sample
from getiprompt.data.utils import read_image

# Initialize GroundedSAM (text-based visual prompting)
model = GroundedSAM(device="cuda")

# Create reference sample with category labels (no masks needed)
ref_sample = Sample(
    categories=["apple"],
    category_ids=[0],
)

# Fit on reference (learns category-to-id mapping)
model.fit(Batch.collate([ref_sample]))

# Predict on target image using text prompts
target_image = read_image("library/tests/assets/fss-1000/images/apple/2.jpg")
target_sample = Sample(image=target_image)
predictions = model.predict(Batch.collate([target_sample]))

# Access results
masks = predictions[0]["pred_masks"]   # Predicted segmentation masks
boxes = predictions[0]["pred_boxes"]   # Detected bounding boxes
labels = predictions[0]["pred_labels"] # Category labels
```

### Customizing Encoder and SAM Models

You can configure Matcher with different encoder and SAM models:

```python
from getiprompt.models import Matcher
from getiprompt.utils.constants import SAMModelName

# Use a lighter model for faster inference
model = Matcher(
    device="cuda",
    encoder_model="dinov3_small",      # Smaller, faster encoder
    sam=SAMModelName.SAM_HQ_TINY,        # Fast SAM HQ TINY model
)

# Use a heavier model for best accuracy
model = Matcher(
    device="cuda",
    encoder_model="dinov3_huge",       # Largest encoder
    sam=SAMModelName.SAM_HQ,       # Large SAM_HQ model
)
```

**Available encoder models:**

| Model | Description |
| ----- | ----------- |
| `dinov3_small` | DINOv3 Small (fastest, lowest memory) |
| `dinov3_small_plus` | DINOv3 Small+ |
| `dinov3_base` | DINOv3 Base (balanced) |
| `dinov3_large` | DINOv3 Large (default, best accuracy) |
| `dinov3_huge` | DINOv3 Huge (highest accuracy, most memory) |

**Available SAM models:**

| Model | Description |
| ----- | ----------- |
| `SAMModelName.SAM_HQ_TINY` | SAM-HQ Tiny (default, fast) |
| `SAMModelName.SAM_HQ` | SAM-HQ (higher quality masks) |
| `SAMModelName.SAM2_TINY` | SAM2 Tiny (newest architecture) |
| `SAMModelName.SAM2_SMALL` | SAM2 Small |
| `SAMModelName.SAM2_BASE` | SAM2 Base |
| `SAMModelName.SAM2_LARGE` | SAM2 Large (highest quality) |

### Using Your Own Images with FolderDataset

Load custom images using `FolderDataset` with this folder structure:

```text
your_dataset/
├── images/
│   ├── category1/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── category2/
│       └── ...
└── masks/
    ├── category1/
    │   ├── 1.png  # Binary mask matching 1.jpg
    │   ├── 2.png
    │   └── ...
    └── category2/
        └── ...
```

```python
from getiprompt.data.folder import FolderDataset
from getiprompt.data.base import Batch

# Load your dataset
dataset = FolderDataset(
    root="path/to/your_dataset",
    categories=["category1", "category2"],  # Or None for all categories
    n_shots=2,  # Number of reference images per category
)

# Get reference and target samples
ref_dataset = dataset.get_reference_dataset()
target_dataset = dataset.get_target_dataset()

# Create batches for model
reference_batch = Batch.collate([ref_dataset[i] for i in range(len(ref_dataset))])
target_batch = Batch.collate([target_dataset[i] for i in range(len(target_dataset))])

# Fit and predict
model.fit(reference_batch)
predictions = model.predict(target_batch)
```

> **Note:** Mask files should be binary images (0 = background, 255 = foreground) with the same filename stem as the corresponding image (e.g., `1.jpg` → `1.png`).

## 🧪 Benchmarking

Evaluate models on standard datasets:

```bash
# Benchmark on LVIS dataset (default)
getiprompt benchmark --dataset_name LVIS --model Matcher

# Benchmark on PerSeg dataset
getiprompt benchmark --dataset_name PerSeg --model Matcher

# Run all models on a dataset
getiprompt benchmark --dataset_name LVIS --model all

# Comprehensive benchmark (all models, all datasets)
getiprompt benchmark --model all --dataset_name all --class_name benchmark
```

> 📊 Results are saved to `~/outputs/` by default.

### Setting Up the LVIS Dataset

To run benchmarks with the LVIS dataset, set up the following folder structure:

```text
~/.cache/getiprompt/datasets/lvis/
├── train2017/
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017/
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── lvis_v1_train.json
└── lvis_v1_val.json
```

**Download COCO images:**

```bash
cd ~/.cache/getiprompt/datasets/lvis

# Download and extract images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip train2017.zip
unzip val2017.zip
```

**Download LVIS annotations:**

Visit the [LVIS Dataset page](https://www.lvisdataset.org/dataset) to download the annotation files, then place them in the root folder.

### Setting Up the PerSeg Dataset

To run benchmarks with the PerSeg dataset, set up the following folder structure:

```text
~/datasets/PerSeg/
├── Images/
│   ├── backpack/
│   │   ├── 00.jpg
│   │   ├── 01.jpg
│   │   └── ...
│   ├── dog/
│   │   └── ...
│   └── ...
└── Annotations/
    ├── backpack/
    │   ├── 00.png
    │   ├── 01.png
    │   └── ...
    ├── dog/
    │   └── ...
    └── ...
```

**Download PerSeg dataset:**

The PerSeg dataset can be downloaded from the [Personalize-SAM repository](https://github.com/ZrrSkywalker/Personalize-SAM).

## 💻 Hardware Requirements

Approximate GPU memory requirements for different model configurations:

| Encoder | SAM Model | GPU Memory |
| ------- | --------- | ---------- |
| `dinov3_small` | `SAM_HQ_TINY` | ~4 GB |
| `dinov3_base` | `SAM_HQ_TINY` | ~6 GB |
| `dinov3_large` | `SAM_HQ_TINY` | ~8 GB |
| `dinov3_large` | `SAM_HQ` | ~10 GB |
| `dinov3_huge` | `SAM_HQ` | ~16 GB |
| `dinov3_huge` | `SAM2_LARGE` | ~20 GB |

> **Note:** Memory usage varies with input image resolution. Values above are for 1024×1024 images.

## 🧮 Supported Models

### Foundation Models (Backbones)

| Family | Models | Description | Paper | Repository |
| ------ | ------ | ----------- | ----- | ---------- |
| **SAM** | SAM-HQ, SAM-HQ-tiny | High-quality variants of the original Segment Anything Model. | [Segment Anything](https://arxiv.org/abs/2304.02643), [SAM-HQ](https://arxiv.org/abs/2306.01567) | [SAM](https://github.com/facebookresearch/segment-anything), [SAM-HQ](https://github.com/SysCV/sam-hq) |
| **SAM 2** | SAM2-tiny, SAM2-small, SAM2-base, SAM2-large | The next generation of Segment Anything, offering improved performance and speed. | [SAM 2](https://arxiv.org/abs/2408.00714) | [sam2](https://github.com/facebookresearch/sam2) |
| **SAM 3** | SAM 3 | Segment Anything with Concepts, supporting open-vocabulary prompts. | [SAM 3](https://arxiv.org/abs/2511.16719) | [SAM 3](https://github.com/facebookresearch/sam3) |
| **MobileSAM** | MobileSAM | Lightweight SAM for mobile applications. | [MobileSAM](https://arxiv.org/abs/2306.14289) | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) |
| **EfficientViT** | EfficientViT-SAM | Accelerated SAM without accuracy loss. | [EfficientViT-SAM](https://arxiv.org/abs/2402.05008) | [EfficientViT](https://github.com/mit-han-lab/efficientvit) |
| **DINOv2** | Small, Base, Large, Giant | Self-supervised vision transformers with registers, used for feature extraction. | [DINOv2](https://arxiv.org/abs/2304.07193), [Registers](https://arxiv.org/abs/2309.16588) | [dinov2](https://github.com/facebookresearch/dinov2) |
| **DINOv3** | Small, Small+, Base, Large, Huge | The latest iteration of DINO models. | [DINOv3](https://arxiv.org/abs/2508.10104) | [dinov3](https://github.com/facebookresearch/dinov3) |
| **Grounding DINO** | (Integrated in GroundedSAM) | Open-set object detection model. | [Grounding DINO](https://arxiv.org/abs/2303.05499) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |

### Visual Prompting Algorithms

| Algorithm | Description | Paper | Repository | Code |
| --------- | ----------- | ----- | ---------- | ---- |
| **Matcher** | Standard feature matching pipeline using SAM. | [Matcher](https://arxiv.org/abs/2305.13310) | [Matcher](https://github.com/aim-uofa/Matcher) | [matcher.py](src/getiprompt/models/matcher/matcher.py) |
| **SoftMatcher** | Enhanced matching pipeline with soft feature comparison, inspired by Optimal Transport. | [IJCAI 2024](https://www.ijcai.org/proceedings/2024/1000.pdf) | N/A | [soft_matcher.py](src/getiprompt/models/soft_matcher.py) |
| **PerDino** | Personalized DINO-based prompting, leveraging DINOv2/v3 features for robust matching. | [PerSAM](https://arxiv.org/abs/2305.03048) | [Personalize-SAM](https://github.com/ZrrSkywalker/Personalize-SAM) | [per_dino.py](src/getiprompt/models/per_dino.py) |
| **GroundedSAM** | Combines Grounding DINO and SAM for text-based visual prompting and segmentation. | [Grounding DINO](https://arxiv.org/abs/2303.05499), [SAM](https://arxiv.org/abs/2304.02643) | [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) | [grounded_sam.py](src/getiprompt/models/grounded_sam.py) |

## ✍️ Acknowledgements

This project builds upon several open-source repositories. See [third-party-programs.txt](../third-party-programs.txt) for the full list.
