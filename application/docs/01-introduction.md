# Geti Prompt - Overview

## Introduction

### What is Geti Prompt?

Geti Prompt is an open-source Python application designed to empower edge users—from domain experts to developers and data scientists—to build computer vision models efficiently. It streamlines the creation, deployment, and management of configurable zero-shot and few-shot AI pipelines. With flexible deployment options, Geti Prompt can be installed as a desktop application or containerized as a Docker image for edge deployment.

Built on the Geti Apps ecosystem, Geti Prompt utilizes a modular, open-source architecture designed for advanced zero-shot learning. It integrates and optimizes foundation models such as Segment Anything and DINO for Intel hardware, enabling single-click deployment of ready-to-use models on industrial edge devices.

![model lifecycle](assets/model-lifecycle.png)

## How It Works

Geti Prompt provides an integrated workflow that seamlessly connects data ingestion with actionable insights:

1. **Connect Input** – Connect your data source directly in the UI, whether it's a webcam, IP camera, GenICam, or a folder of images.

2. **Prompt** – Define what to detect using text descriptions or visual examples captured from your stream.

3. **Test** – See how your prompts perform in real-time and refine them instantly to improve accuracy.

4. **Deploy** – Once your model meets performance requirements, deploy the full application as a containerized Docker image for edge devices, or run it directly on your local PC.

5. **Inference** – Route your results to the right place, such as APIs, message queues, or file systems.

6. **Monitor** – Track real-time performance metrics like throughput and latency to ensure smooth operation.

7. **Act** – Trigger automated workflows or alerts in external systems (like Node-RED) based on model detections.

## Documentation Structure

This documentation is organized into the following sections:

- **[Quick Start](./quick-start.md)**: Installation and first steps
- **[Tutorials](./tutorials/)**: Step-by-step guides for common tasks
- **[How-to Guides](./how-to-guides/)**: In-depth feature documentation
- **[Concepts](./concepts/)**: Technical background and architecture
- **[API Reference](./api-reference.md)**: REST API and Python SDK documentation
- **[FAQ](./faq.md)**: Common questions and troubleshooting

## Next Steps

Start with the [Quick Start Guide](./quick-start.md) to install and run the application, or explore the [Tutorials](./tutorials/) for hands-on learning.
