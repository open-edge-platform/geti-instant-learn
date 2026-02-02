# Geti-Prompt Onboarding Summary

Welcome to **Geti-Prompt** - a full-stack application for Zero/Few-shot Vision Segmentation using visual and text prompts. This document provides an overview of the core library architecture, models, data structures, and workflows.

## Table of Contents
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Core Concepts](#core-concepts)
- [Data Structures](#data-structures)
- [Available Models](#available-models)
- [Model API: fit() and predict()](#model-api-fit-and-predict)
- [Running Models](#running-models)
- [Running Benchmarks](#running-benchmarks)
- [OpenVINO Integration](#openvino-integration)
- [Image Encoders](#image-encoders)
- [SAM Integration](#sam-integration)
- [Development Workflow](#development-workflow)

---

## Installation

The project uses `uv` for package management. Install with the appropriate extras for your hardware:

```bash
cd library

# CPU-only (default for CI/testing)
uv sync --extra cpu --extra dev

# NVIDIA CUDA 12.6
uv sync --extra cu126 --extra dev

# Intel XPU (Arc/Data Center GPUs)
uv sync --extra xpu --extra dev

# With Jupyter notebook support
uv sync --extra cpu --extra dev --extra notebook

# Or use the Justfile (defaults to CPU)
just venv          # Equivalent to: uv sync --extra cpu
```

**Available Extras:**

| Extra | Description |
|-------|-------------|
| `cpu` | PyTorch CPU backend |
| `cu126` | PyTorch with CUDA 12.6 |
| `xpu` | PyTorch with Intel XPU support |
| `dev` | Development tools (ruff, pytest, mypy, pre-commit) |
| `notebook` | Jupyter notebook support |
| `full` | All extras combined |

> **Note:** `cpu`, `cu126`, and `xpu` are mutually exclusive - only install one hardware backend.

---

## Project Overview

Geti-Prompt enables segmentation of objects in images using minimal examples (few-shot learning) or text descriptions (zero-shot). The library is located in `library/src/getiprompt/`.

**Key Directories:**
```
library/src/getiprompt/
├── components/           # Building blocks (encoders, SAM, feature extractors)
│   ├── encoders/         # Image encoders (DINOv2, DINOv3)
│   ├── sam/              # SAM implementations (PyTorch, OpenVINO)
│   ├── prompt_generators/# Point/box prompt generation strategies
│   └── filters/          # Prompt filtering logic
├── data/                 # Dataset handling
│   ├── base/             # Sample, Batch, Dataset base classes
│   ├── folder/           # FolderDataset for custom data
│   ├── lvis/             # LVIS benchmark dataset
│   └── per_seg/          # PerSeg benchmark dataset
├── models/               # Segmentation models
│   ├── matcher/          # Matcher and InferenceMatcher (OpenVINO)
│   ├── base.py           # Model base class
│   ├── soft_matcher.py   # SoftMatcher model
│   ├── per_dino.py       # PerDino model
│   └── grounded_sam.py   # GroundedSAM (text-based)
├── scripts/              # CLI scripts
└── utils/                # Utilities and constants
```

---

## Core Concepts

### Reference vs Target

The fundamental paradigm in Geti-Prompt is **reference** (few-shot examples) vs **target** (images to segment):

| Concept | Description |
|---------|-------------|
| **Reference Sample/Batch** | Images with known object masks that the model learns from |
| **Target Sample/Batch** | Images where the model predicts object locations |
| **is_reference** | Boolean flag per instance indicating if it's a reference |
| **n_shot** | Shot number (0-indexed) for reference instances, -1 for targets |

**Workflow:**
1. **fit()** - Model learns from reference samples (stores embeddings)
2. **predict()** - Model applies learned knowledge to segment targets

---

## Data Structures

### Sample

A `Sample` represents a single image with N object instances. Defined in `data/base/sample.py`.

```python
from getiprompt.data.base.sample import Sample
import numpy as np

sample = Sample(
    # Required
    image=torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8),  # CHW format
    image_path="path/to/image.jpg",
    
    # Annotations (optional, but at least one needed for meaningful tasks)
    masks=torch.randint(0, 2, (N, 224, 224), dtype=torch.uint8),  # (N, H, W)
    bboxes=torch.tensor([[x, y, w, h], ...], dtype=torch.float32),         # (N, 4)
    points=torch.tensor([[x, y], ...], dtype=torch.float32),               # (N, 2)
    
    # Metadata
    categories=["cat", "dog"],                    # N category names
    category_ids=torch.tensor([0, 1], dtype=torch.int32), # N category IDs
    
    # Task-specific
    is_reference=[True, False],  # Per-instance reference flags
    n_shot=[0, -1],              # Per-instance shot numbers
)
```

**Key Points:**
- Images are stored in **CHW (Channels, Height, Width)** format
- Supports **single-instance** (N=1, e.g., PerSeg) and **multi-instance** (N>1, e.g., LVIS)
- All masks have the same H×W dimensions (typically the image size)

### Batch

A `Batch` is a wrapper around `list[Sample]` for batched operations. Defined in `data/base/batch.py`.

```python
from getiprompt.data.base import Batch, Sample

# Create batch from samples
samples = [sample1, sample2, sample3]
batch = Batch.collate(samples)

# Or from a single sample
batch = Batch.collate(sample1)

# Access properties
len(batch)              # Number of samples
batch[0]                # Get first sample
batch.images            # list[tv_tensors.Image] - lazy conversion
batch.masks             # list[torch.Tensor | None]
batch.categories        # list[list[str]]
batch.is_reference      # list[list[bool]]
```

### reference_batch vs target_batch

In model workflows:

```python
# Reference batch: images WITH known masks for learning
reference_batch = Batch(samples=[ref_sample1, ref_sample2])  # is_reference=[True]

# Target batch: images WITHOUT masks (to be predicted)
target_batch = Batch(samples=[target_sample1, target_sample2])  # is_reference=[False]

# Model workflow
model.fit(reference_batch)           # Learn from references
predictions = model.predict(target_batch)  # Predict on targets
```

### Dataset

Base class for all datasets (`data/base/base.py`). Provides:

```python
from getiprompt.data import PerSegDataset, LVISDataset, FolderDataset

# Load a dataset
dataset = PerSegDataset(root="~/datasets/PerSeg", n_shots=1)
dataset = LVISDataset(root="~/datasets/lvis", categories=["cat", "dog"], n_shots=1)
dataset = FolderDataset(root="./my_data", categories=["apple"])

# Split into reference/target
ref_dataset = dataset.get_reference_dataset(category="cat")
target_dataset = dataset.get_target_dataset(category="cat")

# Create batches
reference_batch = Batch(samples=[ref_dataset[i] for i in range(len(ref_dataset))])
```

---

## Available Models

All models inherit from `Model` base class (`models/base.py`) and implement:
- `fit(reference_batch)` - Learn from reference samples
- `predict(target_batch)` - Predict masks on target samples
- `export(export_dir, backend)` - Export to ONNX/OpenVINO

### 1. Matcher (Primary Model)

Based on ICLR'24 paper: "Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"

```python
from getiprompt.models import Matcher

model = Matcher(
    sam=SAMModelName.SAM_HQ_TINY,      # SAM variant
    encoder_model="dinov3_large",      # Image encoder
    num_foreground_points=40,          # Prompt points
    num_background_points=2,
    mask_similarity_threshold=0.38,
    precision="bf16",
    device="cuda",
)
```

**Key Features:**
- Uses **DINOv3** patch encoding instead of SAM encoder for robust features
- **Bidirectional prompt generator** using linear sum assignment
- Complex mask post-processing for filtering and merging

### 2. SoftMatcher (Fast Variant)

Based on IJCAI'24: "Probabilistic Feature Matching for Fast Scalable Visual Prompting"

```python
from getiprompt.models import SoftMatcher

model = SoftMatcher(
    sam=SAMModelName.SAM_HQ_TINY,
    encoder_model="dinov3_large",
    use_sampling=False,
    approximate_matching=False,         # Use Random Fourier Features
    softmatching_score_threshold=0.4,
    softmatching_bidirectional=False,
)
```

**Key Features:**
- Replaces bidirectional matching with **soft matching algorithm**
- Optional **Random Fourier Features** for faster similarity computation
- Better scalability for large images

### 3. PerDino (Grid-Based)

Simpler approach using grid-based prompting.

```python
from getiprompt.models import PerDino

model = PerDino(
    sam=SAMModelName.SAM_HQ_TINY,
    encoder_model="dinov3_large",
    num_grid_cells=16,              # Grid granularity
    similarity_threshold=0.65,      # Matching threshold
)
```

**Key Features:**
- Uses **cosine similarity** for feature matching
- **Grid prompt generator** for multi-object detection
- Simpler but effective baseline

### 4. GroundedSAM (Zero-Shot/Text-Based)

Uses text descriptions instead of visual examples.

```python
from getiprompt.models import GroundedSAM

model = GroundedSAM(
    sam=SAMModelName.SAM_HQ_TINY,
    grounding_model=GroundingModel.LLMDET_TINY,
    box_threshold=0.4,
    text_threshold=0.3,
)

# fit() just stores category mappings
model.fit(reference_batch)

# predict() uses text descriptions
predictions = model.predict(target_batch)
```

**Key Features:**
- Zero-shot object detection using text prompts
- Integrates with Grounding DINO or LLM-Det models
- No visual examples needed

### 5. InferenceMatcher (OpenVINO Optimized)

OpenVINO-based Matcher for efficient CPU/GPU inference.

```python
from getiprompt.models import InferenceMatcher

model = InferenceMatcher(
    model_folder="./exports/matcher_ov",  # Pre-exported models
    sam=SAMModelName.SAM_HQ_TINY,
    device="CPU",  # OpenVINO device
    precision="fp32",
)
```

---

## Model API: fit() and predict()

### fit(reference_batch)

Learns from reference samples by extracting and storing feature embeddings.

```python
def fit(self, reference_batch: Batch) -> None:
    """Learn context from reference samples.
    
    Args:
        reference_batch: Batch of reference samples WITH masks
    """
    # Encode images
    ref_embeddings = self.encoder(reference_batch.images)
    
    # Extract masked features (pooled by mask regions)
    self.masked_ref_embeddings, self.ref_masks = self.masked_feature_extractor(
        ref_embeddings,
        reference_batch.masks,
        reference_batch.category_ids,
    )
```

### predict(target_batch)

Uses learned context to segment target images.

```python
def predict(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
    """Predict masks on target samples.
    
    Args:
        target_batch: Batch of target samples (no masks needed)
        
    Returns:
        list of dicts with predictions per image:
            - "pred_masks": torch.Tensor [num_masks, H, W]
            - "pred_points": torch.Tensor [num_points, 4] (x, y, score, fg_label)
            - "pred_boxes": torch.Tensor [num_boxes, 5] (x1, y1, x2, y2, score)
            - "pred_labels": torch.Tensor [num_masks] (category IDs)
    """
    # Encode targets
    target_embeddings = self.encoder(target_batch.images)
    
    # Generate point prompts from similarity matching
    point_prompts, similarities = self.prompt_generator(
        self.ref_embeddings,
        self.masked_ref_embeddings,
        self.ref_masks,
        target_embeddings,
    )
    
    # Filter prompts
    point_prompts = self.prompt_filter(point_prompts)
    
    # Decode masks using SAM
    return self.segmenter(target_images, point_prompts=point_prompts)
```

---

## Running Models

### Using CLI

```bash
# Run on custom data
getiprompt run --model Matcher \
    --data_root ~/data/my_dataset \
    --output_location ~/outputs \
    --n_shots 1

# Run with text prompts (GroundedSAM)
getiprompt run --model GroundedSAM \
    --text_prompt "cat, dog, bird" \
    --output_location ~/outputs
```

### Programmatic Usage

```python
from getiprompt.models import Matcher
from getiprompt.data import FolderDataset, Batch

# 1. Load dataset
dataset = FolderDataset(root="./my_data", categories=["apple", "orange"])

# 2. Create model
model = Matcher(device="cuda", precision="bf16")

# 3. Get reference samples and fit
ref_dataset = dataset.get_reference_dataset()
reference_batch = Batch.collate([ref_dataset[i] for i in range(len(ref_dataset))])
model.fit(reference_batch)

# 4. Get target samples and predict
target_dataset = dataset.get_target_dataset()
target_batch = Batch.collate([target_dataset[i] for i in range(len(target_dataset))])
predictions = model.predict(target_batch)

# 5. Process results
for i, pred in enumerate(predictions):
    masks = pred["pred_masks"]      # (N, H, W) boolean
    boxes = pred["pred_boxes"]      # (N, 5) with scores
    labels = pred["pred_labels"]    # (N,) category IDs
```

---

## Running Benchmarks

### Using CLI

```bash
# Quick test with defaults
getiprompt benchmark --model Matcher --dataset PerSeg

# Full benchmark
getiprompt benchmark \
    --model Matcher \
    --dataset lvis \
    --class_name benchmark \
    --n_shot 1 \
    --batch_size 4 \
    --experiment_name my_experiment
```

### Category Presets

```python
from getiprompt.scripts.benchmark import load_dataset_by_name

# Preset options: "default", "benchmark", "all"
dataset = load_dataset_by_name("lvis", categories="default")   # 4 categories
dataset = load_dataset_by_name("lvis", categories="benchmark") # 92 categories
dataset = load_dataset_by_name("lvis", categories="all")       # All categories

# Explicit list
dataset = load_dataset_by_name("lvis", categories=["cat", "dog", "bird"])
```

### Programmatic Benchmarking

```python
from getiprompt.scripts.benchmark import perform_benchmark_experiment, load_dataset_by_name
from getiprompt.utils.benchmark import load_model
from getiprompt.utils.constants import ModelName, SAMModelName

# Load components
dataset = load_dataset_by_name("lvis", categories=["cat", "dog"])
model = load_model(sam=SAMModelName.SAM_HQ_TINY, model_name=ModelName.MATCHER, args=args)

# Run benchmark loop
for category in dataset.categories:
    ref_dataset = dataset.get_reference_dataset(category=category)
    reference_batch = Batch.collate(ref_dataset)
    model.fit(reference_batch)
    
    target_dataset = dataset.get_target_dataset(category=category)
    predictions = model.predict(Batch.collate(target_dataset))
    # Compute metrics...
```

---

## OpenVINO Integration

### Exporting Models

```python
from getiprompt.models import Matcher
from getiprompt.utils.constants import Backend
from pathlib import Path

# 1. Create PyTorch model
matcher = Matcher(device="cpu", precision="fp32")

# 2. Export to OpenVINO
matcher.export(
    export_dir=Path("./exports/matcher_ov"),
    backend=Backend.OPENVINO  # or "openvino"
)

# Creates:
# ./exports/matcher_ov/
#   ├── image_encoder.xml  (and .bin)
#   └── exported_sam.xml   (and .bin)
```

### Running with OpenVINO

```python
from getiprompt.models import InferenceMatcher
from getiprompt.utils.constants import SAMModelName

# Load pre-exported model
ov_matcher = InferenceMatcher(
    model_folder="./exports/matcher_ov",
    sam=SAMModelName.SAM_HQ_TINY,
    device="CPU",   # OpenVINO device: "CPU", "GPU", "AUTO"
    precision="fp32",
)

# Same API as PyTorch model
ov_matcher.fit(reference_batch)
predictions = ov_matcher.predict(target_batch)
```

### Export Individual Components

```python
from getiprompt.components.encoders import ImageEncoder
from getiprompt.components.sam import SAMPredictor
from getiprompt.utils.constants import Backend

# Export encoder
encoder = ImageEncoder(model_id="dinov3_large", backend=Backend.TIMM)
encoder.export(Path("./exports"), backend=Backend.OPENVINO)

# Export SAM
predictor = SAMPredictor(sam_model_name=SAMModelName.SAM_HQ_TINY)
predictor.export(Path("./exports"), backend=Backend.OPENVINO)
```

---

## Image Encoders

### Available Encoders

| Model ID | Description | Backend |
|----------|-------------|---------|
| `dinov3_small` | DINOv3 Small | TIMM |
| `dinov3_small_plus` | DINOv3 Small+ | TIMM |
| `dinov3_base` | DINOv3 Base | TIMM |
| `dinov3_large` | DINOv3 Large (default) | TIMM |
| `dinov3_huge` | DINOv3 Huge | TIMM |
| `dinov2_small/base/large/giant` | DINOv2 variants | HuggingFace |

### Using Different Encoders

```python
from getiprompt.components.encoders import ImageEncoder
from getiprompt.utils.constants import Backend

# TIMM backend (DINOv3)
encoder = ImageEncoder(
    model_id="dinov3_large",
    backend=Backend.TIMM,
    device="cuda",
    precision="bf16",
)

# HuggingFace backend (DINOv2)
encoder = ImageEncoder(
    model_id="dinov2_large",
    backend=Backend.HUGGINGFACE,
    device="cuda",
)

# OpenVINO backend (requires pre-exported model)
encoder = ImageEncoder(
    backend=Backend.OPENVINO,
    model_path=Path("./exports/image_encoder.xml"),
    device="CPU",
)

# In models
model = Matcher(encoder_model="dinov3_base")  # Changes backbone
```

---

## SAM Integration

### Available SAM Models

| SAMModelName | Description |
|--------------|-------------|
| `SAM_HQ_TINY` | SAM-HQ Tiny (default, fast) |
| `SAM_HQ` | SAM-HQ ViT-H (large, accurate) |
| `SAM2_TINY` | SAM2 Tiny |
| `SAM2_SMALL` | SAM2 Small |
| `SAM2_BASE` | SAM2 Base+ |
| `SAM2_LARGE` | SAM2 Large |

### Using Different SAM Models

```python
from getiprompt.components.sam import SAMPredictor
from getiprompt.utils.constants import SAMModelName, Backend

# PyTorch backend
predictor = SAMPredictor(
    sam_model_name=SAMModelName.SAM2_BASE,
    backend=Backend.PYTORCH,
    device="cuda",
    precision="bf16",
)

# OpenVINO backend
predictor = SAMPredictor(
    sam_model_name=SAMModelName.SAM_HQ_TINY,
    backend=Backend.OPENVINO,
    model_path=Path("./exports/exported_sam.xml"),
    device="CPU",
)

# In models
model = Matcher(sam=SAMModelName.SAM2_LARGE)  # Changes SAM variant
```

---

## Development Workflow

### Running Tests

```bash
just test-unit          # Unit tests
just test-integration   # Integration tests
just tests              # All tests
```

### Linting & Formatting

```bash
just style-fix    # Auto-fix style issues
just lint         # Run linting checks
```

### Quick Development Pattern

```python
# 1. Test with minimal data
from pathlib import Path
from getiprompt.data import FolderDataset

# Use test assets
dataset = FolderDataset(
    root=Path("examples/assets/fss-1000"),
    categories=["apple"],
)

# 2. Quick model test
from getiprompt.models import Matcher

model = Matcher(device="cpu", precision="fp32")  # CPU for quick testing

# 3. Run fit/predict cycle
ref_batch = Batch.collate(dataset.get_reference_dataset())
model.fit(ref_batch)

target_batch = Batch.collate(dataset.get_target_dataset())
predictions = model.predict(target_batch)
```

---

## Summary: Model Architecture Comparison

| Model | Prompt Type | Encoder | Key Innovation |
|-------|-------------|---------|----------------|
| **Matcher** | Point prompts | DINOv3 | Bidirectional assignment, robust features |
| **SoftMatcher** | Point prompts | DINOv3 | Soft matching, RFF approximation |
| **PerDino** | Point prompts | DINOv3 | Grid-based, cosine similarity |
| **GroundedSAM** | Box prompts | Grounding Model | Text-to-box, zero-shot |

All models share:
- Same `fit()` / `predict()` API
- Same `Sample` / `Batch` data structures
- Same SAM decoder for final mask generation
- OpenVINO export capability

---

## Quick Reference

```python
# Imports
from getiprompt.models import Matcher, SoftMatcher, PerDino, GroundedSAM, InferenceMatcher
from getiprompt.data import Sample, Batch, Dataset, FolderDataset, LVISDataset, PerSegDataset
from getiprompt.components.encoders import ImageEncoder
from getiprompt.components.sam import SAMPredictor
from getiprompt.utils.constants import Backend, SAMModelName, ModelName

# Quick workflow
model = Matcher(device="cuda")
model.fit(reference_batch)
predictions = model.predict(target_batch)
model.export("./exports", backend=Backend.OPENVINO)
```

For more details, explore the example notebooks in `library/examples/` and the test cases in `library/tests/`.

