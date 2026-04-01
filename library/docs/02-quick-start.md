# Quick Start

Get up and running with the Geti Instant Learn Library in minutes.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Installation

```bash
cd library
uv sync --extra xpu    # Intel XPU (recommended)
uv sync --extra cpu    # CPU only
uv sync --extra gpu    # CUDA support
```

Or with pip:

```bash
pip install ./library[xpu]  # or [cpu], [gpu]
```

<details>
<summary><strong>Optional dependencies</strong></summary>

```bash
uv sync --extra dev       # Development tools (ruff, pre-commit)
uv sync --extra notebook  # Jupyter notebook support
uv sync --extra full      # All dependencies
```

</details>

## Your First Inference

### Basic Usage with Matcher

```python
import torch
from instantlearn.models import Matcher
from instantlearn.data import Sample
from instantlearn.components.sam import SAMPredictor
from instantlearn.utils.constants import SAMModelName

# Generate reference mask from a point click using SAM
predictor = SAMPredictor(SAMModelName.SAM_HQ_TINY, device="xpu")
predictor.set_image("examples/assets/coco/000000286874.jpg")
ref_mask, _, _ = predictor.forward(
    point_coords=torch.tensor([[[280, 237]]], device="xpu"),  # Click on elephant
    point_labels=torch.tensor([[1]], device="xpu"),           # 1 = foreground
    multimask_output=False,
)

# Initialize Matcher (device: "xpu", "cuda", or "cpu")
model = Matcher(device="xpu")

# Create reference sample with the generated mask
ref_sample = Sample(
    image_path="examples/assets/coco/000000286874.jpg",
    masks=ref_mask[0],
)

# Fit on reference
model.fit(ref_sample)

# Predict on target image
target_sample = Sample(image_path="examples/assets/coco/000000390341.jpg")
predictions = model.predict(target_sample)

# Access results
masks = predictions[0]["pred_masks"]  # Predicted segmentation masks
```

### Text-Based Prompting with GroundedSAM

```python
from instantlearn.models import GroundedSAM
from instantlearn.data import Sample

# Initialize GroundedSAM (no reference masks needed)
model = GroundedSAM(device="xpu")

# Create reference with category labels only
ref_sample = Sample(categories=["elephant"])

# Fit and predict
model.fit(ref_sample)
target_sample = Sample(image_path="examples/assets/coco/000000390341.jpg")
predictions = model.predict(target_sample)

# Access results
masks = predictions[0]["pred_masks"]
boxes = predictions[0]["pred_boxes"]
labels = predictions[0]["pred_labels"]
```

### Zero-Shot Segmentation with SAM3 OpenVINO

SAM3OpenVINO provides text, box, point, and visual exemplar prompting using
pre-exported OpenVINO IR models — no PyTorch required at inference time.

```python
from instantlearn.models import SAM3OpenVINO
from instantlearn.models.sam3 import SAM3OVVariant
from instantlearn.data import Sample

# Auto-downloads FP16 model from HuggingFace (also supports INT8, INT4, FP32)
model = SAM3OpenVINO(variant=SAM3OVVariant.FP16, device="CPU")

# Text prompt — detect elephants
predictions = model.predict([
    Sample(image_path="examples/assets/coco/000000286874.jpg", categories=["elephant"]),
])
```

<details>
<summary><strong>Visual exemplar mode — fit once, predict on any image</strong></summary>

```python
from instantlearn.models.sam3 import Sam3PromptMode
import numpy as np

model_ve = SAM3OpenVINO(
    variant=SAM3OVVariant.FP16,
    prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
    device="CPU",
)

ref = Sample(
    image_path="examples/assets/coco/000000286874.jpg",
    bboxes=np.array([[180, 105, 490, 370]]),
    categories=["elephant"],
    category_ids=[0],
)
model_ve.fit(ref)
predictions = model_ve.predict([
    Sample(image_path="examples/assets/coco/000000390341.jpg"),
])
```

</details>

See the full set of examples in [sam3_openvino_example.py](../examples/sam3_openvino_example.py).

## CLI Usage

The library provides a command-line interface with three subcommands: `run`, `benchmark`, and `ui`.

### Run a Pipeline

```bash
# Run with predefined masks
instantlearn run \
  --reference_images path/to/reference/images \
  --target_images path/to/target/images \
  --reference_prompts path/to/reference/masks

# Run with text prompt
instantlearn run --target_images path/to/target/images --reference_text_prompt "can"
```

### Benchmark on Datasets

```bash
# Evaluate on LVIS dataset
instantlearn benchmark --dataset_name LVIS --model Matcher

# Evaluate on PerSeg dataset
instantlearn benchmark --dataset_name PerSeg --model Matcher

# Run all models on all datasets
instantlearn benchmark --model all --dataset_name all
```

> Results are saved to `~/outputs/` by default.

## Next Steps

- [Tutorials: Getting Started](tutorials/01-getting-started.md) — Step-by-step walkthrough
- [How-To: Custom Datasets](how-to-guides/01-custom-dataset.md) — Use your own images
- [How-To: Benchmarking](how-to-guides/02-benchmarking.md) — Evaluate model performance
