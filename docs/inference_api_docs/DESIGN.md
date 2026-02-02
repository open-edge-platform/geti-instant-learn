# GetiPrompt Inference API Design

## Overview

GetiPrompt provides a unified API for training, exporting, and deploying vision models with few-shot learning. Models use a two-stage architecture: extract reference embeddings, then predict on targets using those embeddings.

## Core Architecture

### Two-Stage Pattern

Models split into explicit submodules:

```text
Reference Images + Priors → Extractor → Embeddings
Target Images + Embeddings → Predictor → Predictions
```

This separation enables dynamic reference updates and clean export.

### Model Interface

```python
class Model:
    def fit(self, reference_images, reference_priors) -> Results:
        """Extract and store reference embeddings."""

    def predict(self, target_images) -> Results:
        """Predict on targets using stored embeddings."""

    def __call__(self, target_images) -> Results:
        """Support model(input) syntax."""

    def export(self, export_dir: str, mode: str, backend: str) -> Path:
        """Unified export method - delegates to backend-specific methods."""
```

### Submodules

```python
model.extractor   # ReferenceExtractor - extracts embeddings
model.predictor   # TargetPredictor - predicts using embeddings
```

## Export System

### Export Modes

**Static mode** (80% of cases):

- Embeddings baked into model
- Single file, ~400MB
- Fastest inference
- Fixed references

**Dynamic mode** (flexible deployment):

- Extractor + Predictor separate
- 2 files, 1.5-2.8GB
- Change references at runtime
- More memory but flexible

```python
# Static export (requires fit() first)
model.fit(ref_images, ref_priors)
model.export("./exports", mode="static")

# Dynamic export (flexible references)
model.export("./exports", mode="dynamic")
```

### Export API

Unified `export()` method delegates to backend-specific methods:

```python
# Unified export API (recommended)
model.export("./exports", mode="static", backend="tensorrt")
model.export("./exports", mode="dynamic", backend="onnx")

# Backend-specific methods also available
model.to_tensorrt("./exports", mode="static")
model.to_onnx("./exports", mode="dynamic")
model.to_openvino("./exports", mode="static")
```

### Supported Backends

- **ONNX**: Cross-platform baseline
- **TensorRT**: NVIDIA GPUs (2-4x speedup)
- **TorchScript**: PyTorch ecosystem
- **OpenVINO**: Intel CPUs/GPUs/VPUs

## Production Deployment

### InferenceModel Interface

```python
from getiprompt.inference import InferenceModel

# Auto-detects mode and backend from files
model = InferenceModel.load("./exports/matcher")

# Same API as PyTorch models
model.fit(ref_images, ref_priors)  # No-op for static, computes for dynamic
results = model.predict(target_images)
results = model(target_images)  # Or via __call__
```

### Runtime Adapters (Under the Hood)

Backend-specific adapters provide unified interface:

```python
# src/getiprompt/inference/adapters/base.py
class RuntimeAdapter(ABC):
    @abstractmethod
    def load_model(self, path: str) -> Any:
        """Load model from path."""

    @abstractmethod
    def forward(self, inputs: dict) -> dict:
        """Run inference."""

# Concrete implementations
# - OpenVINOAdapter (openvino)
# - ONNXAdapter (onnxruntime)
# - TensorRTAdapter (tensorrt)
# - TorchScriptAdapter (torch.jit)
```

### Backend Detection

Auto-detect backend from file extension:

```python
.onnx → ONNXAdapter
.engine/.plan → TensorRTAdapter
.pt/.pth → TorchScriptAdapter
.xml/.bin → OpenVINOAdapter
```

### InferenceModel Implementation

```python
class InferenceModel:
    """Unified inference interface for all backends."""

    def __init__(self, export_dir: str, backend: str = "auto"):
        self.export_dir = Path(export_dir)
        self.backend = self._detect_backend() if backend == "auto" else backend
        self.adapter = self._get_adapter(self.backend)
        self.mode = self._detect_mode()  # static or dynamic

        if self.mode == "static":
            self.model = self.adapter.load_model(self._get_static_model_path())
        else:
            self.extractor = self.adapter.load_model(self._get_extractor_path())
            self.predictor = self.adapter.load_model(self._get_predictor_path())

    @classmethod
    def load(cls, export_dir: str) -> "InferenceModel":
        """Load exported model with auto-detection."""
        return cls(export_dir, backend="auto")

    def fit(self, ref_images, ref_priors):
        """Extract reference embeddings (for dynamic mode)."""
        if self.mode == "static":
            return  # No-op for static mode
        embeddings = self.adapter.forward(self.extractor, {
            "images": ref_images,
            "priors": ref_priors
        })
        self.embeddings = embeddings

    def predict(self, target_images):
        """Predict on targets - same API as PyTorch."""
        if self.mode == "static":
            return self.adapter.forward(self.model, {"images": target_images})
        return self.adapter.forward(self.predictor, {
            "images": target_images,
            "embeddings": self.embeddings
        })

    def __call__(self, target_images):
        """Support model(input) syntax."""
        return self.predict(target_images)
```

### Key Features

- Auto-detection of export mode (static/dynamic) and backend
- Unified API regardless of export mode
- Dynamic reference updates (for dynamic mode)
- Consistent fit()/predict() interface
- Runtime adapters handle backend differences

## Implementation Structure

### Proposed File Structure

```text
getiprompt/
├── inference/
│   ├── __init__.py
│   ├── model.py                    # InferenceModel class
│   └── adapters/
│       ├── __init__.py
│       ├── base.py                 # RuntimeAdapter interface
│       ├── onnx.py                 # ONNXAdapter
│       ├── tensorrt.py             # TensorRTAdapter
│       ├── torchscript.py          # TorchScriptAdapter
│       └── openvino.py             # OpenVINOAdapter
```

## Workflow

### Development

```python
from getiprompt.models import Matcher

model = Matcher()
model.fit(reference_images, reference_priors)
results = model.predict(target_images)
```

### Export

```python
# Static export
model.fit(ref_images, ref_priors)
model.export("./exports", mode="static")

# Dynamic export
model.export("./exports", mode="dynamic")
```

### Production

```python
from getiprompt.inference import InferenceModel

model = InferenceModel.load("./exports")
model.fit(ref_images, ref_priors)  # Optional for static, required for dynamic
results = model.predict(target_images)
```

## Performance

| Backend | Speedup | Memory |
|---------|---------|--------|
| PyTorch | 1.0x | 2.4 GB |
| ONNX | 1.5-2.0x | 2.0 GB |
| TensorRT | 2.0-4.0x | 1.8 GB |
| OpenVINO | 1.5-2.0x | 2.2 GB |

## Design Principles

1. **Separation of concerns**: Extract vs Predict
2. **Stateless design**: No side effects in exported modules
3. **Backend agnostic**: Unified API across runtimes
4. **Dynamic flexibility**: Change references without re-export
5. **Production ready**: Optimized for real-world deployment

## Export Detection

InferenceModel detects export mode automatically:

- Filename patterns: `*_static.*` → static, `*_extractor.*` → dynamic
- File count: 1 file → static, 2 files → dynamic
- Model introspection: Input/output names reveal structure

No metadata required—works with any export.
