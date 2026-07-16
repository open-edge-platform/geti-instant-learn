# Geti Instant Learn Application

Full-stack web application for deploying zero-shot visual prompting models on live video streams, cameras, and video files.

Built with the [Geti Instant Learn Library](../library/) for model inference, FastAPI for the backend, and React for the frontend.

## License acceptance

By installing, using, or distributing this library/application, you acknowledge that:

- you have read and understood the license terms at the links below;
- you confirmed the linked terms govern the contents you seek to access and use;
- you accepted and agreed to the linked license terms.

License links:

- [SAM3 License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

In order to consent, set an environment variable `INSTANTLEARN_LICENSE_ACCEPTED=1` or accept terms when first importing the library.

## Quick Start

### Hugging Face model access

Some models are gated on Hugging Face. Loading them fails until you have an account, are granted access, and authenticate.

1. Create a Hugging Face account, or sign in: [huggingface.co/join](https://huggingface.co/join)
2. Create a Hugging Face access token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Set the "Read access to contents of all public gated repos you can access" permission for this token.
3. Request access on each model's Hugging Face page:
   - SAM3: [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
   - DINOv3: [huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
4. Use this token on your environment:
   - Windows PowerShell: `[Environment]::SetEnvironmentVariable("HF_TOKEN", "your-token-value", "User")`
   - Linux: `export HF_TOKEN=<your-token-value>`

Retry the model load once access is granted and your token is set.

### Install from source code

The installer clones the repository, sets up its own
copy of `uv`, Node.js and npm under `.build/`, detects your accelerator (Intel® XPU, NVIDIA® CUDA, or CPU), builds the
backend and UI, and starts the app. The first build downloads several GB of packages (PyTorch, OpenVINO, …) and can
take a while — progress is shown for each step.

> [!NOTE]
> `git` is required on all platforms; `curl` is also required on Linux/WSL. Re-running the installer reuses the cached
> tools and dependencies so only the first build is slow.

#### Linux / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/geti-instant-learn/main/install.sh | bash
```

To pass flags — `-v`/`--verbose` (stream full output), `-y`/`--yes` (non-interactive), `-w`/`--work-dir <path>` (custom
install directory, default `./geti-instant-learn`) — forward them through the pipe with `bash -s --`:

```bash
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/geti-instant-learn/main/install.sh | bash -s -- --yes --work-dir ~/geti-instant-learn
```

#### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/open-edge-platform/geti-instant-learn/main/install.ps1 | iex
```

To pass parameters — `-Verbose` (stream full output), `-Yes`/`-y` (non-interactive), `-WorkDir <path>`/`-w` (custom
install directory, default `.\geti-instant-learn`) — run the downloaded script as a script block instead:

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/open-edge-platform/geti-instant-learn/main/install.ps1))) -Yes -WorkDir C:\geti-instant-learn
```

If your execution policy blocks remote scripts, download first and run it explicitly (Bypass applies only to this
process and does not change your machine policy):

```powershell
curl.exe -L https://raw.githubusercontent.com/open-edge-platform/geti-instant-learn/main/install.ps1 -o install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

If a build step fails, re-run with `--verbose` (Linux) or `-Verbose` (Windows), or inspect the log at
`<work-dir>/.build/.install.log`.

### Run from Source (Development)

**Prerequisites:** [uv](https://github.com/astral-sh/uv), [Just](https://github.com/casey/just), Python 3.13, Node.js v24+, [HF token](#hugging-face-model-access)

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
| `device` | `cpu` | Hardware target (`cpu`, `xpu`, `cuda`) |
| `enable-coturn` | `false` | Enable local TURN server for WebRTC |
| `stun-server` | `""` | External STUN server URL |
| `coturn-port` | `443` | Port for local TURN server |

> **Note:** WebRTC parameters configure video streaming between browser and backend. See [WebRTC Networking](docs/04-concepts/02-webrtc.md) for deployment scenarios.

</details>

### Run with Docker

**Prerequisites:** [Just](https://github.com/casey/just), Docker, [HF token](#hugging-face-model-access)

**Build the image:**

```bash
# Build for Intel XPU (recommended)
just device=xpu application/build-image
```

<details>
<summary><b>Build parameters</b></summary>

| Variable | Default | Description |
| :--- | :--- | :--- |
| `device` | `cpu` | Hardware target: `cpu`, `xpu`, `cuda` |
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
| `device` | `cpu` | Hardware target: `cpu`, `xpu`, `cuda` |
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
