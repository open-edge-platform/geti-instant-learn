# Geti Prompt Library

A flexible and modular framework for exploring, developing, and evaluating visual prompting algorithms.

---

## What is Visual Prompting?

Imagine you need to locate and precisely outline every instance of a specific object in hundreds of images—perhaps "kettles" in kitchen photos or "tumors" in medical scans. Traditional approaches typically require either:

1. Training a dedicated model on thousands of labeled examples, or
2. Manually segmenting each image (time-consuming and labor-intensive)

**Visual prompting** offers a powerful alternative: show the model just _one or a few examples_ of what you're looking for, and it can find and segment similar objects in new images.

At its core, visual prompting uses:

- **Reference images** containing examples of your target object (with corresponding masks)
- **Feature matching** that compares visual patterns between reference and target images
- **Guided segmentation** that uses these matches to precisely outline objects of interest

This approach is valuable when you have limited labeled data and want to quickly prototype object detection for new categories. Performance is strongest when target objects share visual characteristics with the model's training data. For specialized domains (medical imaging, satellite imagery), domain-specific fine-tuning may improve results.

## Key Features

- **Modular Pipeline Architecture** — Mix and match components (backbones, feature extractors, matchers) to create custom pipelines
- **Extensive Algorithm Support** — Matcher, SoftMatcher, PerDino, GroundedSAM, and more
- **Comprehensive Evaluation** — Unified benchmarking with support for LVIS, PerSeg, and standard metrics
- **Easy Integration** — Simple API for adding new algorithms, backbones, or datasets

## Example

```python
from getiprompt.models import Matcher
from getiprompt.data import Sample

# Use example assets from the library (apple images from FSS-1000)
ref_image = "examples/assets/fss-1000/images/apple/1.jpg"
ref_mask = "examples/assets/fss-1000/masks/apple/1.png"
target_image = "examples/assets/fss-1000/images/apple/2.jpg"

# Initialize Matcher (device: "xpu", "cuda", or "cpu")
model = Matcher(device="xpu")
model.fit(Sample(image_path=ref_image, mask_paths=[ref_mask]))
predictions = model.predict(Sample(image_path=target_image))
```

## Supported Models

### Visual Prompting Algorithms

| Algorithm | Description |
| --------- | ----------- |
| **Matcher** | Standard feature matching pipeline using SAM |
| **SoftMatcher** | Enhanced matching with soft feature comparison (Optimal Transport) |
| **PerDino** | Personalized DINO-based prompting with DINOv2/v3 features |
| **GroundedSAM** | Text-based visual prompting combining Grounding DINO and SAM |

### Foundation Models

| Family | Models |
| ------ | ------ |
| **SAM** | SAM-HQ, SAM-HQ-tiny |
| **SAM 2** | SAM2-tiny, SAM2-small, SAM2-base, SAM2-large |
| **SAM 3** | Segment Anything with Concepts (open-vocabulary) |
| **DINOv2** | Small, Base, Large, Giant |
| **DINOv3** | Small, Small+, Base, Large, Huge |

## Next Steps

- [Quick Start Guide](02-quick-start.md) — Installation and first inference
- [Concepts](concepts/01-concepts.md) — Core concepts and architecture
- [Tutorials](tutorials/01-getting-started.md) — Step-by-step guides

## Acknowledgements

This project builds upon several open-source repositories. See the [third-party-programs.txt](https://github.com/open-edge-platform/geti-prompt/blob/main/third-party-programs.txt) file for the full list.
