# Quick Start Guide

Get your first zero-shot segmentation results in minutes.

## Installation

> **Note:** Pre-built binaries are not yet available. This section will be updated when container images are published to the registry. For now, build and run the application using Docker as described in the [Application README](https://github.com/open-edge-platform/instant-learn/blob/main/application/README.md).

Once running, access the application at `http://localhost:9100`

## Create Your First Project

1. **Create a project:** Click "Create Project" on the welcome screen
2. **Add input:** Select a sample dataset from the dropdown
3. **Capture an image:** Click through the dataset to find a good example
4. **Draw your prompt:** Use the annotation tool to select target objects and assign a label (e.g., coffee berries)
5. **Apply the prompt:** Click "Save Prompt" to instruct the model
6. **View results:** Navigate through the dataset to see zero-shot predictions

The default PerDINO model runs automatically on all images using your visual prompt.

## Recommended Hardware Setups

For optimal performance with zero-shot visual prompting models:

- **Intel GPU** — Best performance for real-time inference

**Pre-configured systems:**

- [Edge AI Systems Catalog](https://builders.intel.com/ecosystem-engagement/solution-hub/edge-ai-catalog/partner-spotlight?cp=35%2CArray&cid=37&type=system) — Certified edge AI systems from Intel partners

**Build your own:**

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| CPU | Intel Core Ultra (Series 1 or 2) | Intel Core Ultra 7 or 9 |
| GPU | Intel Arc GPU (integrated) | Intel Arc Pro B580 or discrete Arc GPU |
| RAM | 8 GB | 16 GB |
| OS | Linux (Docker) or Windows 11 (MSIX) | Ubuntu 24.04 or Windows 11 |

> **Note:** OpenVINO backend provides optimized CPU inference. Intel XPU recommended for real-time performance.

## Recommended Hardware Specifications

**Intel XPU Options:**

- **[Intel Arc GPU for Edge](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc/edge.html)** — Discrete GPUs optimized for edge AI workloads
- **[Built-in Intel Arc Pro GPU](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/mobile/overview.html)** — Integrated graphics in mobile processors
- **[Intel Arc Pro B-Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html)** — Professional graphics cards for workstations
