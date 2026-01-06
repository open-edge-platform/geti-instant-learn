![alt text](../assets/geti-prompt-header.png)

<div align="center">

**A flexible and modular framework for visual prompting algorithms**

---

[![python](https://img.shields.io/badge/python-3.12%2B-green)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](../LICENSE)

</div>

# 👋 Introduction

The Geti Prompt Library provides a robust platform for experimenting with visual prompting techniques. Its modular pipeline design allows researchers and developers to easily combine, swap, and extend components such as backbone networks, feature extractors, matching algorithms, and mask generators.

# 📦 Installation

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

# 🚀 Quick Start

## Python API

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
ref_image = read_image("tests/assets/fss-1000/images/apple/1.jpg")

# Initialize SAM predictor (auto-downloads weights)
predictor = PyTorchSAMPredictor(SAMModelName.SAM_HQ_TINY, device="cuda")

# Set image and generate mask from a point click
predictor.set_image(ref_image.unsqueeze(0), original_size=ref_image.shape[1:])
ref_mask, _, _ = predictor.predict(
    point_coords=torch.tensor([[[150, 150]]], device="cuda"),  # Click on apple
    point_labels=torch.tensor([[1]], device="cuda"),           # 1 = foreground
    multimask_output=False,
)
```

**Fit and predict with Matcher:**

```python
from getiprompt.models import Matcher
from getiprompt.data import Batch, Sample

# Initialize model
model = Matcher(device="cuda")

# Create reference sample with the generated mask
ref_sample = Sample(
    image=ref_image,
    masks=ref_mask,
    categories=["apple"],
)
model.fit(Batch.collate([ref_sample]))

# Predict on target image
target_image = read_image("tests/assets/fss-1000/images/apple/2.jpg")
target_sample = Sample(image=target_image)
predictions = model.predict(Batch.collate([target_sample]))

# Access results
masks = predictions[0]["pred_masks"]   # Shape: [N, H, W]
boxes = predictions[0]["pred_boxes"]   # Shape: [N, 5] (x1, y1, x2, y2, score)
labels = predictions[0]["pred_labels"] # Shape: [N]
```

## Command Line

```bash
# Run with predefined masks
getiprompt run \
    --reference_images path/to/reference \
    --target_images path/to/target \
    --reference_prompts path/to/masks

# Run with text prompt (zero-shot)
getiprompt run \
    --target_images path/to/target \
    --reference_text_prompt "can"

# Use different model and backbone
getiprompt run --pipeline SoftMatcher --pipeline.sam MOBILE_SAM ...
```

# 🧪 Benchmarking

Evaluate models on standard datasets:

```bash
# Quick benchmark on LVIS
getiprompt benchmark

# Specify dataset and model
getiprompt benchmark --dataset_name PerSeg --model Matcher

# Run all models
getiprompt benchmark --model all

# Comprehensive benchmark
getiprompt benchmark --model all --dataset_name all --class_name benchmark
```

> 📊 Results are saved to `~/outputs/` by default.

# 🧮 Supported Models

## Foundation Models (Backbones)

| Family | Models | Description | Paper | Repository |
|--------|--------|-------------|-------|------------|
| **SAM** | SAM-HQ, SAM-HQ-tiny | High-quality variants of the original Segment Anything Model. | [Segment Anything](https://arxiv.org/abs/2304.02643), [SAM-HQ](https://arxiv.org/abs/2306.01567) | [SAM](https://github.com/facebookresearch/segment-anything), [SAM-HQ](https://github.com/SysCV/sam-hq) |
| **SAM 2** | SAM2-tiny, SAM2-small, SAM2-base, SAM2-large | The next generation of Segment Anything, offering improved performance and speed. | [SAM 2](https://arxiv.org/abs/2408.00714) | [sam2](https://github.com/facebookresearch/sam2) |
| **SAM 3** | SAM 3 | Segment Anything with Concepts, supporting open-vocabulary prompts. | [SAM 3](https://arxiv.org/abs/2511.16719) | [SAM 3](https://github.com/facebookresearch/sam3) |
| **MobileSAM** | MobileSAM | Lightweight SAM for mobile applications. | [MobileSAM](https://arxiv.org/abs/2306.14289) | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) |
| **EfficientViT** | EfficientViT-SAM | Accelerated SAM without accuracy loss. | [EfficientViT-SAM](https://arxiv.org/abs/2402.05008) | [EfficientViT](https://github.com/mit-han-lab/efficientvit) |
| **DINOv2** | Small, Base, Large, Giant | Self-supervised vision transformers with registers, used for feature extraction. | [DINOv2](https://arxiv.org/abs/2304.07193), [Registers](https://arxiv.org/abs/2309.16588) | [dinov2](https://github.com/facebookresearch/dinov2) |
| **DINOv3** | Small, Small+, Base, Large, Huge | The latest iteration of DINO models. | [DINOv3](https://arxiv.org/abs/2508.10104) | [dinov3](https://github.com/facebookresearch/dinov3) |
| **Grounding DINO** | (Integrated in GroundedSAM) | Open-set object detection model. | [Grounding DINO](https://arxiv.org/abs/2303.05499) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |

## Visual Prompting Algorithms

| Algorithm | Description | Paper | Repository |
|-----------|-------------|-------|------------|
| **Matcher** | Standard feature matching pipeline using SAM. | [Matcher](https://arxiv.org/abs/2305.13310) | [Matcher](https://github.com/aim-uofa/Matcher) |
| **SoftMatcher** | Enhanced matching pipeline with soft feature comparison, inspired by Optimal Transport. | [IJCAI 2024](https://www.ijcai.org/proceedings/2024/1000.pdf) | N/A |
| **PerDino** | Personalized DINO-based prompting, leveraging DINOv2/v3 features for robust matching. | [PerSAM](https://arxiv.org/abs/2305.03048) | [Personalize-SAM](https://github.com/ZrrSkywalker/Personalize-SAM) |
| **GroundedSAM** | Combines Grounding DINO and SAM for text-based visual prompting and segmentation. | [Grounding DINO](https://arxiv.org/abs/2303.05499), [SAM](https://arxiv.org/abs/2304.02643) | [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) |

# 📚 Documentation

For detailed documentation on datasets, advanced usage, and API reference, see [docs/01-introduction.md](docs/01-introduction.md).

# ✍️ Acknowledgements

This project builds upon several open-source repositories. See [third-party-programs.txt](../third-party-programs.txt) for the full list.
