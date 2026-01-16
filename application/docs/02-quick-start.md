# Quick Start Guide

Get up and running with Geti Prompt in minutes.

## 1. Installation

You can run Geti Prompt directly from source (for development) or as a container (for isolation).

### Option A: Run from Source (Recommended for Devs)

**Prerequisites:** [uv](https://github.com/astral-sh/uv), [Just](https://github.com/casey/just), Python 3.12+, Node.js v24+.

```bash
# Start backend and frontend in dev mode
just application/dev

# Recommended: Run on Intel XPU (GPU/NPU)
just device=xpu application/dev

# Optional: Run on NVIDIA GPU
just device=cu126 application/dev
```

**Access the application at: [http://localhost:3000](http://localhost:3000)**

### Option B: Run with Docker

**Prerequisites:** [Just](https://github.com/casey/just), Docker.

```bash
# Build and run the container automatically
just application/run-image

# Recommended: Run on Intel XPU (GPU/NPU)
just device=xpu application/run-image

# Optional: Run on NVIDIA GPU
just device=cu126 application/run-image
```

**Access the application at: [http://localhost:9100](http://localhost:9100)**

---

## 2. Configuration

You can customize the startup command with these variables.

| Variable | Default | Description | Example |
| :--- | :--- | :--- | :--- |
| `port` | `9100` | Backend API port | `just port=8080 ...` |
| `ui-port` | `3000` | UI port (Source mode only) | `just ui-port=4000 ...` |
| `device` | `cpu` | Hardware (`cpu`, `xpu`, `cu126`) | `just device=xpu ...` |
| `enable-coturn` | `false` | Enable local Coturn TURN server | `just enable-coturn=true ...` |
| `coturn-port` | `443` | Port for Coturn server | `just coturn-port=3478 ...` |
| `stun-server` | `""` | External STUN server URL | `just stun-server="stun:..."` |
| `webrtc-ports` | `50000-51000` | UDP port range for WebRTC | `just webrtc-ports="50000-50100"` |

> **Note**: For detailed explanation of WebRTC parameters and when to use them, see [WebRTC Networking](concepts/02-development.md#webrtc-networking).

## 3. Basic Usage

### Step 1: Configure Input

Navigate to **Settings > Input** and select your video source (USB Camera, IP Camera, Video File, etc.).

### Step 2: Create a Valid Prompt

1. Click **Capture** to grab a reference frame.
2. Select an object in the image using the prompting tools.
3. Assign a label and save.

### Step 3: View Results

Inference starts immediately. View real-time results in **Live View** or configure data export in **Settings > Output**.

> **Note**: Model weights are downloaded automatically when the model is deployed for the first time. Please be patient waiting for inference results after prompting a model.

## Next Steps

- **[Tutorials](tutorials/01-tutorials.md)**: Step-by-step guides for specific use cases.
- **[How-to Guides](how-to-guides/01-how-to-guides.md)**: Deep dive into features.
- **[Concepts](concepts/01-concepts.md)**: Learn about the architecture.
