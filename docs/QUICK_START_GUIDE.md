# Quick Start Guide: Running Application Backend and UI

This guide covers the configuration changes and commands needed to run the Geti Prompt application locally.

## Prerequisites

- Python with `uv` package manager
- Node.js v24.2.0
- CUDA (if using GPU)

## Configuration Changes

### 1. Update CORS Origins in Backend Settings

Edit `application/backend/app/settings.py`:

```python
cors_origins: str = Field(
    default="http://192.168.86.41:3000, http://192.168.86.41:9100",
    alias="CORS_ORIGINS",
)
```

Replace `192.168.86.41` with your workstation's IP address.

### 2. Configure AI Extra for CUDA (GPU Only)

Edit `application/backend/Justfile`:

```just
ai-extra := "cu126"  # Use cu126 for CUDA support
```

### 3. Set AI Device

Edit `application/Justfile`:

```just
ai-device := "gpu"  # options: cpu, gpu, xpu
```

### 4. Configure Runtime for CUDA

Edit `application/backend/app/runtime/core/components/factories/model.py`:

```python
runtime = os.getenv("RUNTIME", "cuda").lower()
```

## Running the Application

### Start Backend

```bash
cd application/backend
just dev
```

### Start UI

```bash
cd application/ui
npm install
npm start
```

> **Note**: Ensure Node.js version is v24.2.0. Use `nvm` to switch versions if needed:
> ```bash
> nvm use 24.2.0
> ```

## Verification

1. Backend should be running on the configured port
2. UI should be accessible at `http://<workstation-ip>:3000`
3. Check browser console for any CORS errors

## Troubleshooting

- **CORS errors**: Verify the IP address in `settings.py` matches your workstation IP
- **CUDA not found**: Ensure `ai-extra` is set to the correct CUDA version (`cu126`)
- **Node version mismatch**: Run `node --version` to verify v24.2.0
