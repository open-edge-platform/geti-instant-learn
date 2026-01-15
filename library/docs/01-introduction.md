# Geti Prompt Library

**A flexible and modular framework for exploring, developing, and evaluating visual prompting algorithms.**

---

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/open-edge-platform/geti-prompt/blob/main/LICENSE) [![Python Version](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/downloads/) [![version](https://img.shields.io/badge/version-0.1.0-lightgrey.svg)](https://github.com/open-edge-platform/geti-prompt/blob/main/library/pyproject.toml)

This project provides a robust platform for experimenting with various visual prompting techniques. Its core strength lies in a highly **modular pipeline design**, allowing researchers and developers to easily combine, swap, and extend different components such as backbone networks, feature extractors, matching algorithms, and mask generators.

## What is Visual Prompting?

Imagine you need to locate and precisely outline every instance of a specific object in hundreds of images—perhaps "kettles" in kitchen photos or "tumors" in medical scans. Traditional approaches typically require either:

1. Training a dedicated model on thousands of labeled examples of that specific object, or
2. Manually segmenting each image (extremely time-consuming and labor-intensive)

**Visual prompting** offers a powerful alternative: show the model just _one or a few examples_ of what you're looking for, and it can find and segment similar objects in new images.

At its core, visual prompting is a form of few-shot learning for vision models like SAM (Segment Anything Model). It works by using:

- **Reference images** containing examples of your target object (often with corresponding masks)
- **Feature matching** that compares visual patterns between your reference and target images
- **Guided segmentation** that uses these matches to precisely outline the objects of interest

This approach is particularly valuable when you need to segment rare or novel object categories not well-represented in general training data, or when you can't collect large training datasets due to privacy, time, or resource constraints.

This repository explores algorithms that make visual prompting more effective, efficient, and adaptable to different use cases.

## Key Features

- 🧩 **Modular Pipeline Architecture:** Easily configure pipelines by modifying their Python class definitions to mix and match components (backbones, feature extractors, matchers, etc.). Simplifies experimentation and development of novel approaches.
- 🔬 **Extensive Algorithm & Backbone Support:** Includes implementations for various state-of-the-art algorithms (Matcher, SoftMatcher, Dino-based methods) and diverse backbone models (SAM, MobileSAM, DinoV2).
- 📊 **Comprehensive Evaluation Framework:** Unified evaluation script with support for multiple datasets (LVIS, PerSeg, etc.) and standard metrics (mIoU, Precision, Recall).
- 💻 **Interactive Web UI:** Visually inspect similarity maps, generated masks, and points for qualitative analysis and debugging. Easily switch between configurations.
- 🔌 **Easy Integration:** Designed for straightforward addition of new algorithms, backbones, or datasets.

## Installation

Create a new environment and install dependencies. We recommend using `uv` for faster environment creation.

```bash
uv sync

# Install extras (xFormers)
uv sync --extra extras

# Install dev dependencies (ruff, pre-commit)
uv sync --extra dev

# Install all dependencies
uv sync --extra full
```

## Usage

The project is structured around a command-line interface (CLI) with three subcommands: `run`, `benchmark`, and `ui`.

You can get help for any command by using the `-h` or `--help` flag, for example, `getiprompt benchmark --help`.

### Run a Pipeline

The `run` subcommand allows you to execute a pipeline on your own set of images.

**Example:**

```bash
# Run using predefined masks with default pipeline (Matcher)
getiprompt run --reference_images path/to/reference/images --target_images path/to/target/images --reference_prompts path/to/reference/masks

# Run using points
getiprompt run --reference_images path/to/reference/images --target_images path/to/target/images --points "[0:[640,640], -1:[200,200]]"

# Run using text prompt
getiprompt run --target_images path/to/target/images --reference_text_prompt "can"

# Run any other subclass of Pipeline (e.g. SoftMatcher) using MobileSAM
getiprompt run --pipeline SoftMatcher --pipeline.sam MOBILE_SAM ...
```

You can configure the pipeline's parameters using dot notation (e.g., `--pipeline.mask_similarity_threshold`, `--pipeline.device`).

### Benchmark on Datasets

The `benchmark` subcommand is used to evaluate pipeline performance on various datasets. It can be used to test multiple pipelines, datasets, and hyperparameters all in one experiment.

#### Dataset Setup

Before running benchmarks, ensure your datasets are properly structured. By default, datasets are expected in `~/datasets/`.

**PerSeg Dataset Structure:**

```text
~/datasets/PerSeg/
├── Images/
│   ├── backpack/
│   │   ├── 00.jpg
│   │   ├── 01.jpg
│   │   └── ...
│   ├── dog/
│   │   └── ...
│   └── [other_categories]/
└── Annotations/
    ├── backpack/
    │   ├── 00.png
    │   ├── 01.png
    │   └── ...
    ├── dog/
    │   └── ...
    └── [other_categories]/
```

**LVIS Dataset Structure:**

```text
~/datasets/lvis/
├── train2017/          # COCO train images
│   ├── 000000000001.jpg
│   └── ...
├── val2017/            # COCO val images
│   ├── 000000000001.jpg
│   └── ...
├── lvis_v1_train.json  # LVIS annotations
└── lvis_v1_val.json    # LVIS annotations
```

> **Note:** LVIS uses COCO images. Download from [COCO](https://cocodataset.org/#download) and [LVIS](https://www.lvisdataset.org/dataset).

You can specify a custom dataset root using `--dataset_root`:

```bash
getiprompt benchmark --dataset_root /path/to/datasets
```

#### Basic Usage

```bash
# Evaluate the default model on LVIS with default categories (quick test)
getiprompt benchmark

# Specify dataset and model
getiprompt benchmark --dataset_name PerSeg --model Matcher

# Change number of reference shots
getiprompt benchmark --n_shot 3

# Select a different backbone
getiprompt benchmark --sam MobileSAM

# Run all available models
getiprompt benchmark --model all

# Run all datasets
getiprompt benchmark --dataset_name all
```

> **Available Models:** `Matcher`, `SoftMatcher`, `PerDino`, `GroundedSAM`
> **Available SAM Backbones:** `SAM`, `MobileSAM`, `SAM-HQ`, `SAM-HQ-tiny`, `SAM2-tiny`, `SAM2-small`, `SAM2-base`, `SAM2-large`
> **Available Datasets:** `PerSeg`, `lvis`
>
> **Tip:** Use `all` with `--model`, `--sam`, or `--dataset_name` to run all available options. These can be combined for comprehensive benchmarking.

#### Category Filtering

The benchmark supports three ways to filter categories:

1. **Preset modes** - Predefined category sets for different use cases:

   ```bash
   # Quick testing with default categories (4 categories for LVIS)
   getiprompt benchmark --class_name default

   # Comprehensive benchmark (~100 categories for LVIS)
   getiprompt benchmark --class_name benchmark

   # All available categories in the dataset
   getiprompt benchmark --class_name all
   ```

2. **Explicit category list** - Specify categories directly:

   ```bash
   # Single category
   getiprompt benchmark --class_name cat

   # Multiple categories (comma-separated)
   getiprompt benchmark --class_name cat,dog,bird
   ```

3. **Default behavior** - If `--class_name` is not specified, uses the "default" preset (4 categories for LVIS, all categories for PerSeg).

#### Complete Examples

```bash
# Combine multiple arguments
getiprompt benchmark --dataset_name PerSeg --model SoftMatcher --n_shot 3 --sam MobileSAM --class_name backpack,dog

# Run comprehensive benchmark on LVIS
getiprompt benchmark --dataset_name lvis --class_name benchmark --n_shot 1

# Run all models on all datasets with default categories
getiprompt benchmark --model all --dataset_name all

# Run all models on all datasets with all categories (full benchmark)
getiprompt benchmark --model all --dataset_name all --class_name all

# Custom dataset location with benchmark categories
getiprompt benchmark --dataset_root /custom/path --class_name benchmark
```

See [`src/getiprompt/utils/args.py`](https://github.com/open-edge-platform/geti-prompt/blob/main/library/src/getiprompt/utils/args.py) or run `getiprompt benchmark --help` for all available command-line options. Results (metrics and visualizations) are saved to `~/outputs/` by default.

## Acknowledgements

This project builds upon and utilizes code from several excellent open-source repositories. We thank the authors for their contributions. A full list of third party software can be found in the [third-party-programs.txt](https://github.com/open-edge-platform/geti-prompt/blob/main/third-party-programs.txt) file.
