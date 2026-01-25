# Geti Prompt Application

Full-stack web application for deploying zero-shot visual prompting models on live video streams, cameras, and video files.

Built with the [Geti Prompt Library](../library/) for model inference, FastAPI for the backend, and React for the frontend.

## Quick Start

### Run from Source (Development)

**Prerequisites:** [uv](https://github.com/astral-sh/uv), [Just](https://github.com/casey/just), Python 3.12+, Node.js v24+

```bash
# Start backend and frontend in development mode
just device=xpu application/dev
```

**Access at: [http://localhost:3000](http://localhost:3000)**

<details>
<summary><b>Configuration parameters</b></summary>

| Variable | Default | Description |
| :--- | :--- | :--- |
| `port` | `9100` | Backend API port |
| `ui-port` | `3000` | UI development server port |
| `device` | `cpu` | Hardware target (`cpu`, `xpu`, `cu126`) |
| `enable-coturn` | `false` | Enable local TURN server for WebRTC |
| `stun-server` | `""` | External STUN server URL |
| `coturn-port` | `443` | Port for local TURN server |

</details>

### Run with Docker

**Prerequisites:** [Just](https://github.com/casey/just), Docker

**Build the image:**

```bash
# Build for Intel XPU (recommended)
just device=xpu application/build-image
```

<details>
<summary><b>Build parameters</b></summary>

| Variable | Default | Description |
| :--- | :--- | :--- |
| `device` | `cpu` | Hardware target: `cpu`, `xpu`, `cu126` |
| `build-target` | `cpu` | Docker build stage: `cpu`, `xpu`, `gpu` |
| `container-registry` | `localhost:5000/...` | Registry URL |
| `version` | `latest` | Image version tag |

</details>

**Run the image:**

```bash
# Run with default settings
just device=xpu application/run-image
```

**Access at: [http://localhost:9100](http://localhost:9100)**

<details>
<summary><b>Runtime parameters</b></summary>

*Networking:*

| Variable | Default | Description |
| :--- | :--- | :--- |
| `port` | `9100` | Backend API port |
| `webrtc-ports` | `50000-51000` | UDP port range for WebRTC |
| `stun-server` | `""` | External STUN server URL |
| `enable-coturn` | `false` | Enable local TURN server |
| `coturn-port` | `443` | Port for TURN server |

*Hardware:*

| Variable | Default | Description |
| :--- | :--- | :--- |
| `device` | `cpu` | Hardware target: `cpu`, `xpu`, `cu126` |
| `webcam-device` | `/dev/video0` | Path to webcam device |

</details>

---

## 📖 Documentation

- **[Quick Start Guide](docs/02-quick-start.md)** - Get your first results quickly
- **[Tutorials](docs/tutorials/01-tutorials.md)** - Step-by-step guides
- **[How-to Guides](docs/how-to-guides/01-how-to-guides.md)** - Feature deep dives
- **[Concepts & Architecture](docs/concepts/01-concepts.md)** - System design and core concepts
- **[WebRTC Networking](docs/concepts/02-webrtc.md)** - Configure video streaming for different deployment scenarios
