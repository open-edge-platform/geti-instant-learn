# Geti Prompt Application

Full-stack web application for deploying zero-shot visual prompting models on live video streams, cameras, and video files.

Built with the [Geti Prompt Library](../library/) for model inference, FastAPI for the backend, and React for the frontend.

## Quick Start

### Run from Source (Development)

**Prerequisites:** [uv](https://github.com/astral-sh/uv), [Just](https://github.com/casey/just), Python 3.12+, Node.js v24+

```bash
# Start backend and frontend in development mode
just application/dev

# Recommended: Run on Intel XPU (GPU/NPU)
just device=xpu application/dev

# Optional: Run on NVIDIA GPU
just device=cu126 application/dev
```

**Access at: [http://localhost:3000](http://localhost:3000)**

### Run with Docker

**Prerequisites:** [Just](https://github.com/casey/just), Docker

```bash
# Build and run container
just application/run-image

# Recommended: run on Intel XPU
just device=xpu application/run-image

# Run on NVIDIA GPU
just device=cu126 application/run-image
```

**Access at: [http://localhost:9100](http://localhost:9100)**

---

## Configuration

Customize startup with these variables:

| Variable | Default | Description | Run mode |
| :--- | :--- | :--- | :--- |
| `port` | `9100` | Backend API port | All |
| `ui-port` | `3000` | UI development server port | Source only |
| `device` | `cpu` | Hardware target (`cpu`, `xpu`, `cu126`) | All |
| `enable-coturn` | `false` | Enable local TURN server for WebRTC | All |
| `stun-server` | `""` | External STUN server URL | All |
| `webrtc-ports` | `50000-51000` | UDP port range for WebRTC | All |
| `coturn-port` | `443` | Port for local TURN server | All |
| `webcam-device` | `/dev/video0` | Path to webcam device | Docker only |

**Examples:**

```bash
# Source mode with custom ports
just port=8080 ui-port=4000 device=xpu application/dev

# Docker with webcam passthrough
just webcam-device="/dev/video1" application/run-image
```

---

## 📖 Documentation

- **[Quick Start Guide](docs/02-quick-start.md)** - Detailed installation and configuration
- **[Tutorials](docs/tutorials/01-tutorials.md)** - Step-by-step guides
- **[How-to Guides](docs/how-to-guides/01-how-to-guides.md)** - Feature deep dives
- **[Concepts & Architecture](docs/concepts/01-concepts.md)** - System design and core concepts
- **[Development Guide](docs/concepts/02-development.md)** - Architecture layers, testing, and WebRTC networking
