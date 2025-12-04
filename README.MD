![alt text](assets/geti-prompt-header.png)

[![python](https://img.shields.io/badge/python-3.12%2B-green)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

# 👋 Geti-Prompt

A production-ready platform for Visual Prompting on live video streams.

Geti Prompt bridges the gap between research and production. It is a comprehensive platform that enables users to explore, develop, and deploy visual prompting algorithms. Whether you are experimenting with new foundation models or deploying them for real-time inference, Geti Prompt provides a modular architecture extensible to various streaming sources, designed to support inputs such as IP cameras and GenICams.

## 💡 What is Visual Prompting?

Visual prompting offers a powerful alternative to traditional training. Instead of curating thousands of labeled images, you simply show the model one or a few examples of what you are looking for. The model effectively "learns" instantly, detecting and segmenting similar objects in new images or live video streams without retraining.

## 🚀 Key Features

<details>
<summary>⚡ Live Video Inference</summary>
Process real-time video streams efficiently. Connect to extensible sources including IP cameras, GenICams, and local image folders with low-latency performance.
</details>

<details>
<summary>🧠 Advanced Visual Prompting</summary>
Interact with images using points or bounding boxes. Leverage state-of-the-art models like SAM 3 and DinoV3 to generate precise segmentation masks instantly.
</details>

<details>
<summary>💾 Modular I/O Architecture</summary>
Flexible "Source" and "Sink" system. Configure inputs and direct inference results to the destination of your choice (disk, API, network stream) via a plugin-like architecture.
</details>

<details>
<summary>🧩 Customizable Pipelines</summary>
The core logic is encapsulated in the getiprompt library. Developers can mix and match components (backbones, feature extractors, matchers) by simply modifying Python class definitions.
</details>
<details>
<summary>🖥️ Interactive UI</summary>
A modern React interface for real-time visual prompting. Seamlessly capture frames, annotate objects with a click, and instantly validate model performance on live video streams.
</details>

## 🧮 Supported models

Geti Prompt supports a variety of foundation models and visual prompting algorithms, optimized for different performance needs.

### Foundation Models (Backbones)

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

### Visual Prompting Algorithms

| Algorithm | Description | Paper | Repository |
|-----------|-------------|-------|------------|
| **Matcher** | Standard feature matching pipeline using SAM. | [Matcher](https://arxiv.org/abs/2305.13310) | [Matcher](https://github.com/aim-uofa/Matcher) |
| **SoftMatcher** | Enhanced matching pipeline with soft feature comparison, inspired by Optimal Transport. | [IJCAI 2024](https://www.ijcai.org/proceedings/2024/1000.pdf) | N/A |
| **PerDino** | Personalized DINO-based prompting, leveraging DINOv2/v3 features for robust matching. | [PerSAM](https://arxiv.org/abs/2305.03048) | [Personalize-SAM](https://github.com/ZrrSkywalker/Personalize-SAM) |
| **GroundedSAM** | Combines Grounding DINO and SAM for text-based visual prompting and segmentation. | [Grounding DINO](https://arxiv.org/abs/2303.05499), [SAM](https://arxiv.org/abs/2304.02643) | [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) |

## 🛫 Getting Started

Geti Prompt can be used in two ways: as a **Python library** for research and algorithmic development, or as a **Full Application** for visual prompting with a user interface.

### Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (Python package manager)
- **[Just](https://github.com/casey/just)** (Command runner)
- **Node.js (v24.2.0)** (Required only for the UI Application)

### 1. Using the Library (Research & Development)

Best for developers and researchers who want to experiment with visual prompting algorithms, run benchmarks, or integrate the `getiprompt` package into their own code.

- **Documentation:** [Library Documentation](library/docs/01-introduction.md)
- **Quick Install:**
  ```bash
    # todo: add instructions for the library
  ```

### 2. Running the Application (UI & Backend)

Best for users who want an end-to-end production-ready platform with a web interface to interact with live video streams and visual prompts. Start with a live model from scratch and deploy to production seamlessly. The application features an Intel hardware-optimized inference engines, ensuring high performance on industrial hardware.

- **Documentation:** [Application Documentation](application/docs/02-quick-start.md)
- **Quick Start:**
  ```bash
  # Clone the repository
  git clone https://github.com/open-edge-platform/geti-prompt.git
  cd geti-prompt

  # Run the full application (Backend + UI)
  # todo: add instructions to run the app from source code
  ```
  This will start the backend server (port 9100) and the UI development server.

## 🏗️ High-level architecture

The project is structured as follows:

- **`library/`**: Contains the `getiprompt` Python package.
- **`application/backend/`**: Contains the inference runtime and API service.
- **`application/ui/`**: Contains the React frontend application.

## 🎡 Community

- To report a bug or submit a feature request, please open a [GitHub issue](https://github.com/open-edge-platform/geti-prompt/issues).

## 🙌 Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

## Acknowledgements
This project incorporates code from several open-source repositories. We thank the authors for their contributions. A complete list of third-party software is available in the [third-party-programs.txt](third-party-programs.txt) file.


## 📝 License

Geti Prompt is licensed under the [Apache License 2.0](LICENSE).
