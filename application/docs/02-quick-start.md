# Quick Start Guide

Get up and running with Geti Prompt Application in minutes.

## Installation

Geti Prompt can be used the way you want: as a desktop application on your pc or local workstation or installed as a Docker image on your edge server which you can access via a web browser.

### Option 1: Desktop Application

```bash
# TODO add instructions how to install using an msix package.

```

### Option 2: Docker

```bash
# TODO add instructions how to run the app as a docker container.

```

### Option 3: Run from source code

```bash
# TODO add instructions how to run the app from source code.
```

## Start using Geti Prompt

### Step 1. Input Configuration

To begin using the application, configure the input stream for the application to process.

>TODO: Insert a step-by-step guide with screenshots/GIFs demonstrating how to select and configure input sources (e.g., IP Camera, Web Camera, local video file).

### Step 2. Prompting Models

Geti Prompt offers two interaction modes: Visual Prompting and Text Prompting. You can switch between these methods dynamically in the UI to suit your specific use case. Each mode features its own dedicated pipelines and configurable models.

Geti Prompt is built on the VisionPrompt framework, supporting the latest open-vocabulary foundation models. These state-of-the-art models were selected for their superior benchmarking performance and permissive licensing, allowing for unrestricted deployment

> TODO: include what pipelines are supported for visual prompt & text prompt, how pipelines are configured, where to find more info about the pipeline benchmarks and characteristics, etc. Finish by redirecting user to dedicated “VisionPrompt framework” section.

#### Visual prompting
Visual Prompting allows you to teach the model by directly selecting objects of interest. To create a prompt, simply capture a reference image, select the target object with a single click, and save the prompt. The model uses this visual reference to immediately detect similar objects in your input stream."

- Capture Image: Grab a reference frame directly from the video source.
- Select: Use the built-in tools to select the object.
- Define: assign a class name to the object (e.g., "defect", "person").
- Save Prompt: Commit the prompt to immediately trigger inference on the live stream.
- Manage Prompts: Optimize your inference results by curating and refining the list of active prompts.

> TODO:
> - Add a GIF showing the capture process and best practices for selecting clear reference frames.
> - Add a GIF showing label creation.
> - Explanation of how modifying prompts affects real-time inference.

#### Text prompting
Text Prompting allows you to instruct the model by simply entering a text query in the UI. Just describe the object you want to detect, and the model will interpret your request.

> TODO: include how to prompt by text, what the difference is with visual prompting, how to manage prompts, etc – short gif

### Step 3. Inference & Deployment

1. **Live Visualization:** Upon activating a prompt, Geti Prompt processes the visual input stream in real-time. Detection results are immediately displayed as an overlay in the Live View, allowing you to validate model performance instantly.

2. **Output Configuration:** Customize how inference results are exported and utilized for downstream applications.
- **Destination:** Choose where to send the data (e.g., Local Disk, Network Stream, API Endpoint).
- **Format: Select:** the data structure for predictions (e.g., JSON, CSV).
- **Rate: Control:** the inference frequency (FPS) to manage load.

> TODO: include the different output destination & prediction options, with gif to show how this works and how the stored output looks like as an example

3. **Production Deployment:** Once validated, you can deploy the fully configured application to your target environment. Geti Prompt supports flexible deployment options:
- **Edge/Remote:** Deploy as a containerized Docker image for remote edge servers and devices.
- **Local:** Run the application and model directly on a local PC for testing or desktop usage.
> TODO: include steps for each deployment option, and gifs/screenshots where applicable

### Step 4. Monitoring & Observability
Geti Prompt exposes real-time inference statistics, enabling you to track model health and latency during active deployment.

> TODO: include the different inference performance statistics, how to interpret, how to configure, etc, with gif


## Automation & Integration

Transform raw inference results into actionable workflows. You can pipe model outputs into custom logic flows to automate decision-making.
- **Node-RED Integration:** We provide boilerplate flows for common use cases, allowing you to visually build logic without deep coding.
- **Standard Use Cases:**
    - Counting: Track the total number of detected objects over time.
    - Filtering: Trigger actions only when specific labels or confidence scores are met.

> TODO: include boilerplate flows by using Node-RED for standard use cases like counting, measuring polygon/bounding box size and orientation, filtering labels or confidence thresholds

## Next Steps

- **Learn by Example**: Explore the [Tutorials](tutorials/01-tutorials.md) for specific workflows
- **Deep Dive**: Read the [How-to Guides](how-to-guides/01-how-to-guides.md) for feature documentation
- **Understand the Architecture**: Review the [Concepts](concepts/01-concepts.md) section

## Troubleshooting

For common issues, see the [FAQ](03-faq.md) or the [Development Guide](concepts/02-development.md).
