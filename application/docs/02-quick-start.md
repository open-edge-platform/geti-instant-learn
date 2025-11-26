# Quick Start Guide

Get up and running with Geti Prompt Application in minutes.

## Installation

Geti Prompt can be used the way you want: as a desktop application on your pc or local workstation or installed as a Docker image on your edge server which you can access via a web browser.

### Option 1: Desktop Application

```bash
#

```

### Option 2: Docker

```bash
# Build and run using Docker

```

### Option 3: Run from source code

```bash
# Start the application
# See application README for specific startup instructions
```
## Start using Geti Prompt

### Input Configuration

To begin using the application, configure the input stream for the application to process.
>TODO: Insert a step-by-step guide with screenshots/GIFs demonstrating how to select and configure input sources (e.g., IP Camera, Web Camera, local video file).

### Prompting Models
Geti Prompt provides users with two ways to prompt a model: using visual prompting and text prompting. Depending on the use case and performance requirements, you can select the prompting approach of choice at any time in the UI. Each prompting method comes with its own set of pipelines, i.e. models, that can be configured.

#### Models
Geti Prompt is built upon its own VisionPrompt framework, which supports the latest state-of-the-art foundation models from the literature. These foundation models have been selected based on their benchmarking performance as well as their permissive licenses, so that they can be deployed by users without restrictions.
> TODO: include what pipelines are supported for visual prompt & text prompt, how pipelines are configured, where to find more info about the pipeline benchmarks and characteristics, etc. Finish by redirecting user to dedicated “VisionPrompt framework” section.

#### Visual prompting
For visual prompting, users will be able to teach the model what it should be looking for by highlighting the object(s) of interest on an image. You can manually capture one or few reference images, define the label that correspond to the object that needs to be detected, and annotate the object accordingly to create a prompt. Based on this prompt, the model will then infer on the input data to try to detect the object of interest.

- **Capture Image:** Capture a reference frame directly from the input stream.
- **Define Labels:** Create a class label for the object (e.g., "defect," "person").
- **Annotate:** Use the annotation tools to select the object.
- **Save prompt:** This creates the prompt immediately used by the model for inference.
- **Manage Prompts:** TODO

> TODO:
> - Add a GIF showing the capture process and best practices for selecting clear reference frames.
> - Add a GIF showing label creation.
> - Explanation of how modifying prompts affects real-time inference.


#### Text prompting
For text prompting, users will be able to teach the model what it should be looking for by simply providing a query in the UI.
> TODO: include how to prompt by text, what the difference is with visual prompting, how to manage prompts, etc – short gif

### Inference

- **Live View:** Once a model is prompted, Geti Prompt processes the visual input stream immediately. Results appear in the Live View overlay.
- **Output configuration:** Configure how the generated model output is stored or consumed.
    - **Destination:** Select where data is sent (e.g., local disk, network stream, API endpoint).
    - **Format:** Choose the prediction format (JSON, CSV).
    - **Rate:** Define the inference frequency (FPS).
> TODO: include the different output destination & prediction options, with gif to show how this works and how the stored output looks like as an example]

### Deployment
Once the prompted model meets the user’s performance requirements, the user can deploy the Geti Prompt application, including the prompted model, in different ways: as a containerized Docker image for deployment on remote edge servers and devices, or run the application & model directly on a local pc.
> TODO: include steps for each deployment option, and gifs/screenshots where applicable

### Monitoring
Geti Prompt provides inference statistics in real-time to inform the user about the model performance during deployment.
> TODO: include the different inference performance statistics, how to interpret, how to configure, etc, with gif

### Business logic implementation
The generated output from the deployed model can be consumed by custom business logic flows built by users, so that action can be taken upon the inference results in an automated way.
> TODO: include boilerplate flows by using Node-RED for standard use cases like counting, measuring polygon/bounding box size and orientation, filtering labels or confidence thresholds

## Next Steps

- **Learn by Example**: Explore the [Tutorials](../tutorials/) for specific workflows
- **Deep Dive**: Read the [How-to Guides](../how-to-guides/) for feature documentation
- **Understand the Architecture**: Review the [Concepts](../concepts/) section

## Troubleshooting

For common issues, see the [FAQ](../faq.md) or the [DEVELOPMENT.md](../DEVELOPMENT.md) file.
