# Architecture

Technical background and architectural concepts to help developers and advanced users understand Geti Prompt Application.

## Visual Prompting & Zero-Shot Learning

The application relies on **Zero-Shot Learning (ZSL)** models, which allows it to detect objects that it hasn't been explicitly trained on. Instead of traditional training with thousands of labeled images, the user provides a "prompt" that describes the object of interest. For more details see the [library documentation](https://github.com/open-edge-platform/geti-prompt/blob/main/library/docs/01-introduction.md)

## Pipelines

The core of the application is the **Pipeline**, a multi-threaded streaming engine that orchestrates the flow of data from ingestion to inference and finally to output. It follows a classic **Source-Processor-Sink** pattern, where each component runs in its own thread to ensure non-blocking execution.

<div align="center">

<img src="media/concepts.svg" width="600px" alt="Diagram showing the sequence of processing stages">
</div>

Data flows through the pipeline via **Broadcasters**, thread-safe intermediaries that allow multiple consumers to receive frames simultaneously. This architecture allows:

- **High Performance**: Heavy inference tasks don't block frame acquisition.
- **Real-time Visualization**: The frontend (via WebRTC) can subscribe to the processed video stream without slowing down the pipeline.

## Pipeline Components

The application abstracts input and output devices to make the system flexible and extensible.

- **Sources (Input)**: Ingests frames into the pipeline. A Source wraps a `Reader`—an abstraction over the actual input device (camera, video file, image folder). The Source calls the Reader's `read()` method to obtain frames, then broadcasts them to a queue where the Processor picks them up.

  **Flow Control Modes** determine how the Source invokes the Reader:

  - **Continuous Mode**: For infinite streams (IP cameras, webcams). The Source continuously calls `read()` in a loop.
  - **Manual Mode**: For navigable datasets (image folders). The Source waits for an explicit "next" signal before calling `read()`.

  The application includes standard Reader implementations: `UsbCameraReader`, `VideoStreamReader`, `ImageFolderReader`. Users can extend the application by implementing their own `StreamReader`.

- **Processor (Inference)**: Performs inference on incoming frames. The Processor subscribes to the Source's broadcast queue, pulls frames, batches them for efficiency, and runs them through the Zero-Shot Learning model. After inference, it broadcasts the results (original frame + predictions) to another queue where the Sink and WebRTC frontend pick them up.

  The Processor wraps a `ModelHandler`—an abstraction over the inference backend. The ModelHandler's `predict()` method takes a batch of frames and returns predictions (masks, scores, labels).

  The application supports two backends:

  - **PyTorch**: Standard deep learning backend.
  - **[OpenVINO™](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)**: For hardware-optimized deployment on Intel CPUs, GPUs, and NPUs.

  Users can extend the application by implementing their own `ModelHandler`.

- **Sinks (Output)**: Exports inference results to external systems. A Sink subscribes to the Processor's broadcast queue, pulls results, and passes them to a `Writer`—an abstraction over the output destination. The Writer's `write()` method handles the actual export.

  The application includes a standard Writer implementation: `MqttWriter` for publishing results to an MQTT broker.

    Users can extend the application by implementing their own `StreamWriter` to support custom destinations (databases, PLCs, local files).

## System Architecture

The application is structured as a layered edge server. Each layer has a clear responsibility, from user interaction down to hardware execution.

<div align="center">

<img src="media/system-architecture.svg" width="1200px" alt="Diagram showing the application system architecture">
</div>

### API Layer

The REST API allows users to interact with the system: create projects, configure sources and sinks, select models, and define prompts. All configuration is persisted in an **SQLite** database, ensuring state survives restarts.

### Pipeline Manager

A non-terminating background loop that keeps the runtime in sync with the stored configuration. When a user changes settings via the API, the Pipeline Manager detects the change and updates the running pipeline—replacing components as needed without restarting the application.

### Pipeline

The runtime heart of the system. As described above, it consists of Source, Processor, and Sink components running in separate threads, connected via broadcast queues. This is where frames flow and inference happens.

### Accelerator Framework

The Processor delegates inference to a backend framework:

- **[OpenVINO™](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)**: Graph optimizations, quantization, and support for CPUs, GPUs.
- **PyTorch (XPU)**: Hardware acceleration on Intel GPUs via XPU device support.

### Hardware

The application runs on edge devices with CPU, GPU compute. The accelerator framework abstracts the hardware, allowing the same pipeline to execute on different platforms.

See [Recommended Hardware Specifications](https://github.com/open-edge-platform/geti-prompt/blob/main/application/docs/02-quick-start.md#recommended-hardware-specifications) for detailed hardware requirements.
