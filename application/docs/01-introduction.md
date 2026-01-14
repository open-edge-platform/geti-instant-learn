# Introduction to the Application

This documentation covers the **Geti Prompt Application**, a full-stack solution (UI + Backend) that wraps the core library into a deployable, user-friendly platform.

Focus is specifically on the **Application experience**—how to use the interface to solve real-world computer vision problems without writing code.

---

## Built for Operations and Deployment

The Geti Prompt Application is designed for users who need to deploy computer vision solutions in the real world. It wraps the powerful algorithms of the core library into a robust platform focused on **live inference** and **edge integration**.

* **Visual Interface**: Interact with state-of-the-art models through a web UI. No Python knowledge required.
* **Real-World Data**: Connect directly to IP cameras, USB webcams, and GenICam devices instead of static datasets.
* **Actionable Insights**: Stream inference results to external systems (MQTT, REST APIs) to trigger real-world actions.
* **Hardware Optimized**: Built-in support for Intel® OpenVINO™ ensures your models run efficiently on industrial edge hardware.

## Interactive Workflow

The Application transforms the complex capabilities of foundation models (like SAM 2 and DINOv3) into a simple three-step workflow:

### 1. Connect & Visualize

Instead of dealing with file paths and scripts, you connect directly to live sensors. The application handles the complexity of video decoding, buffering, and stream management, giving you a real-time view of your environment.

### 2. Interactive Prompting

"Training" a model becomes an interactive conversation with the AI:

* **Visual Prompting:** Simply click on an object in the video feed. The app captures that visual signature and tracks it across future frames.
* **Text Prompting (Coming Soon):** Type what you are looking for (e.g., "safety vest").
* **Instant Feedback:** See the results immediately overlaid on the video. If the model makes a mistake, correct it with another click.

### 3. Deploy & Integrate

Once your prompts are working, the Application acts as an edge server. It runs continuously, optimized for Intel hardware (using OpenVINO™), and broadcasts the detection results to your other systems (PLCs, Dashboards, or Databases).

## Architecture at a Glance

The Application is built on a modern stack designed for performance and extensibility:

* **Frontend**: A React-based Single Page Application (SPA) that provides the interactive canvas for video and prompting.
* **Backend**: A FastAPI service that manages the model lifecycle, video pipelines, and hardware acceleration.
* **Communication**: Uses WebSockets for low-latency video streaming and real-time inference updates.

## Next Steps

* **[Quick Start Guide](02-quick-start.md)**: Spin up the application locally or via Docker.
* **[Tutorials](tutorials/01-tutorials.md)**: Walk through a complete end-to-end example.
* **[Concepts](concepts/01-concepts.md)**: Understand the underlying technology (Pipelines, Sources, Sinks).
