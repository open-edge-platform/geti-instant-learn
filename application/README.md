# Geti Instant Learn Application

Full-stack web application for deploying zero-shot visual prompting models on live video streams, cameras, and video files.

Built with the [Geti Instant Learn Library](../library/) for model inference, FastAPI for the backend, and React for the frontend.

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
| `device` | `cpu` | Hardware target (`cpu`, `xpu`, `cu128`) |
| `enable-coturn` | `false` | Enable local TURN server for WebRTC |
| `stun-server` | `""` | External STUN server URL |
| `coturn-port` | `443` | Port for local TURN server |

> **Note:** WebRTC parameters configure video streaming between browser and backend. See [WebRTC Networking](docs/04-concepts/02-webrtc.md) for deployment scenarios.

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
| `device` | `cpu` | Hardware target: `cpu`, `xpu`, `cu128` |
| `build-target` | `cpu` | Docker build stage: `cpu`, `xpu`, `cuda` |
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
| `port` | `9100` | Port for serving UI and API |
| `webrtc-ports` | `50000-51000` | UDP port range for WebRTC |
| `stun-server` | `""` | External STUN server URL |
| `enable-coturn` | `false` | Enable local TURN server |
| `coturn-port` | `443` | Port for TURN server |

> **Note:** WebRTC parameters configure video streaming between browser and backend. See [WebRTC Networking](docs/04-concepts/02-webrtc.md) for deployment scenarios.

*Hardware:*

| Variable | Default | Description |
| :--- | :--- | :--- |
| `device` | `cpu` | Hardware target: `cpu`, `xpu`, `cu128` |
| `webcam-device` | `/dev/video0` | Path to webcam device |

</details>

---

## Documentation

**Getting Started:**

- [Quick Start Guide](docs/02-quick-start.md) - Get your first results quickly

**Using Geti Instant Learn:**

- [Inputs Configuration](docs/03-use-instant-learn/01-inputs-configuration.md) - Configure cameras, videos, and datasets
- [Prompt & Models](docs/03-use-instant-learn/02-prompt-model.md) - Visual and text prompting
- [Inference](docs/03-use-instant-learn/03-inference.md) - Run zero-shot inference
- [Deployment](docs/03-use-instant-learn/04-deployment.md) - Production deployment
- [Monitoring](docs/03-use-instant-learn/05-monitoring.md) - Monitor application performance
- [Integration](docs/03-use-instant-learn/06-integration.md) - Business logic integration

**Concepts:**

- [Architecture](docs/04-concepts/01-architecture.md) - System design and components
- [WebRTC Networking](docs/04-concepts/02-webrtc.md) - Video streaming configuration
- [Storage](docs/04-concepts/03-storage.md) - Data persistence and Docker volumes
