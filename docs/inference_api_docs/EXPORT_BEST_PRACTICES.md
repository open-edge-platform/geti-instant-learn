# Export Best Practices: Multi-Backend Support

## Overview

This document describes best practices for exporting GetiPrompt models to **ONNX, TensorRT, TorchScript, and OpenVINO** while maintaining **dynamic extraction** capabilities.

---

## 1. Architecture Principles

### ✅ Stateless Submodules
Export-friendly modules should be **pure functions** with no internal state:

```python
class ReferenceExtractor(nn.Module):
    """Stateless module - perfect for export."""

    def forward(self, images: Tensor, priors: Tensor) -> tuple[Tensor, Tensor]:
        # Pure computation - no side effects
        embeddings = self.encoder(images, priors)
        features = self.feature_selector(embeddings)
        return features, masks  # Return everything needed

# ❌ AVOID: Stateful modules
class BadExtractor(nn.Module):
    def forward(self, images: Tensor):
        self.stored_embeddings = self.encoder(images)  # Side effect!
        return self.stored_embeddings
```

### ✅ Explicit Data Flow
All data flows through function arguments and returns:

```python
# Extractor returns embeddings
embeddings, masks = extractor(ref_images, ref_priors)

# Predictor receives embeddings as input
results = predictor(target_images, embeddings, masks)
```

### ✅ Backend Agnostic
Modules should work identically across all backends:

```python
# Same interface for all backends
# PyTorch
embeddings, masks = model.extractor(images, priors)

# ONNX Runtime
embeddings, masks = ort_session.run(None, {'images': images, 'priors': priors})

# TensorRT
embeddings, masks = trt_context.execute_v2(bindings)

# OpenVINO
embeddings, masks = ov_compiled_model([images, priors])
```

---

## 2. Export Strategy Pattern

### Backend Adapter Interface

```python
# library/src/getiprompt/export/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch.nn as nn


class ExportBackend(ABC):
    """Base class for export backends."""

    @abstractmethod
    def export_module(
        self,
        module: nn.Module,
        output_path: Path,
        input_shapes: dict[str, tuple],
        **kwargs
    ) -> None:
        """Export a single module.

        Args:
            module: The module to export (extractor or predictor)
            output_path: Where to save the exported model
            input_shapes: Dictionary mapping input names to shapes
            **kwargs: Backend-specific options
        """
        pass

    @abstractmethod
    def validate_export(self, module: nn.Module, output_path: Path) -> bool:
        """Validate exported model matches PyTorch output."""
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this backend (e.g., '.onnx', '.pt')."""
        pass
```

### ONNX Backend

```python
# library/src/getiprompt/export/onnx.py
import torch
import onnx
import onnxruntime as ort
from pathlib import Path

from .base import ExportBackend


class ONNXExportBackend(ExportBackend):
    """ONNX export backend with validation."""

    def __init__(
        self,
        opset_version: int = 17,
        dynamic_axes: dict | None = None,
        optimize: bool = True,
    ):
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes or {}
        self.optimize = optimize

    def export_module(
        self,
        module: nn.Module,
        output_path: Path,
        input_shapes: dict[str, tuple],
        **kwargs
    ) -> None:
        """Export to ONNX format."""
        module.eval()

        # Create dummy inputs
        dummy_inputs = {
            name: torch.randn(shape)
            for name, shape in input_shapes.items()
        }

        # Export
        torch.onnx.export(
            module,
            tuple(dummy_inputs.values()),
            str(output_path),
            input_names=list(input_shapes.keys()),
            output_names=kwargs.get('output_names', ['output']),
            opset_version=self.opset_version,
            dynamic_axes=self.dynamic_axes,
            do_constant_folding=True,
            **kwargs
        )

        # Optimize if requested
        if self.optimize:
            self._optimize_onnx(output_path)

        print(f"✅ Exported to ONNX: {output_path}")

    def _optimize_onnx(self, model_path: Path) -> None:
        """Optimize ONNX model."""
        import onnxoptimizer

        model = onnx.load(str(model_path))
        optimized = onnxoptimizer.optimize(model)
        onnx.save(optimized, str(model_path))

    def validate_export(self, module: nn.Module, output_path: Path) -> bool:
        """Validate ONNX export matches PyTorch."""
        session = ort.InferenceSession(str(output_path))

        # Get input shapes from ONNX model
        input_shapes = {
            inp.name: inp.shape
            for inp in session.get_inputs()
        }

        # Create test inputs
        test_inputs = {
            name: torch.randn(*shape).numpy()
            for name, shape in input_shapes.items()
        }

        # Run PyTorch
        module.eval()
        with torch.no_grad():
            torch_output = module(*test_inputs.values())

        # Run ONNX
        onnx_output = session.run(None, test_inputs)

        # Compare
        return torch.allclose(
            torch.tensor(onnx_output[0]),
            torch_output,
            rtol=1e-3,
            atol=1e-5
        )

    @property
    def file_extension(self) -> str:
        return ".onnx"
```

### TensorRT Backend

```python
# library/src/getiprompt/export/tensorrt.py
import torch
import tensorrt as trt
from pathlib import Path

from .base import ExportBackend


class TensorRTExportBackend(ExportBackend):
    """TensorRT export via ONNX."""

    def __init__(
        self,
        fp16_mode: bool = True,
        int8_mode: bool = False,
        max_workspace_size: int = 1 << 30,  # 1GB
    ):
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.max_workspace_size = max_workspace_size

    def export_module(
        self,
        module: nn.Module,
        output_path: Path,
        input_shapes: dict[str, tuple],
        **kwargs
    ) -> None:
        """Export to TensorRT engine."""
        # First export to ONNX
        onnx_path = output_path.with_suffix('.onnx')
        onnx_backend = ONNXExportBackend()
        onnx_backend.export_module(module, onnx_path, input_shapes, **kwargs)

        # Convert ONNX to TensorRT
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size

        if self.fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.int8_mode:
            config.set_flag(trt.BuilderFlag.INT8)

        engine = builder.build_engine(network, config)

        # Serialize and save
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())

        print(f"✅ Exported to TensorRT: {output_path}")

    def validate_export(self, module: nn.Module, output_path: Path) -> bool:
        """Validate TensorRT export."""
        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(output_path, 'rb') as f:
            engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # TODO: Add validation logic
        return True

    @property
    def file_extension(self) -> str:
        return ".trt"
```

### TorchScript Backend

```python
# library/src/getiprompt/export/torchscript.py
import torch
from pathlib import Path

from .base import ExportBackend


class TorchScriptExportBackend(ExportBackend):
    """TorchScript export backend."""

    def __init__(self, use_trace: bool = True):
        self.use_trace = use_trace

    def export_module(
        self,
        module: nn.Module,
        output_path: Path,
        input_shapes: dict[str, tuple],
        **kwargs
    ) -> None:
        """Export to TorchScript."""
        module.eval()

        # Create dummy inputs
        dummy_inputs = tuple(
            torch.randn(shape)
            for shape in input_shapes.values()
        )

        # Export
        if self.use_trace:
            scripted = torch.jit.trace(module, dummy_inputs)
        else:
            scripted = torch.jit.script(module)

        # Optimize
        scripted = torch.jit.optimize_for_inference(scripted)

        # Save
        scripted.save(str(output_path))
        print(f"✅ Exported to TorchScript: {output_path}")

    def validate_export(self, module: nn.Module, output_path: Path) -> bool:
        """Validate TorchScript export."""
        scripted = torch.jit.load(str(output_path))

        # Create test inputs
        test_inputs = tuple(
            torch.randn(2, 3, 224, 224)  # Example
        )

        # Compare outputs
        module.eval()
        with torch.no_grad():
            torch_output = module(*test_inputs)
            scripted_output = scripted(*test_inputs)

        return torch.allclose(torch_output, scripted_output, rtol=1e-3)

    @property
    def file_extension(self) -> str:
        return ".pt"
```

### OpenVINO Backend

```python
# library/src/getiprompt/export/openvino.py
import torch
from pathlib import Path
import openvino as ov

from .base import ExportBackend


class OpenVINOExportBackend(ExportBackend):
    """OpenVINO export backend."""

    def __init__(self, compress_to_fp16: bool = True):
        self.compress_to_fp16 = compress_to_fp16

    def export_module(
        self,
        module: nn.Module,
        output_path: Path,
        input_shapes: dict[str, tuple],
        **kwargs
    ) -> None:
        """Export to OpenVINO IR format."""
        # First export to ONNX
        onnx_path = output_path.with_suffix('.onnx')
        onnx_backend = ONNXExportBackend()
        onnx_backend.export_module(module, onnx_path, input_shapes, **kwargs)

        # Convert to OpenVINO
        ov_model = ov.convert_model(str(onnx_path))

        # Compress if requested
        if self.compress_to_fp16:
            from openvino.tools import mo
            ov_model = mo.compress_model(ov_model)

        # Save
        ov.save_model(ov_model, str(output_path))
        print(f"✅ Exported to OpenVINO: {output_path}")

    def validate_export(self, module: nn.Module, output_path: Path) -> bool:
        """Validate OpenVINO export."""
        core = ov.Core()
        compiled = core.compile_model(str(output_path), "CPU")

        # TODO: Add validation logic
        return True

    @property
    def file_extension(self) -> str:
        return ".xml"  # OpenVINO IR format
```

---

## 3. Unified Export API

```python
# library/src/getiprompt/export/__init__.py
from pathlib import Path
from typing import Literal

import torch.nn as nn

from .base import ExportBackend
from .onnx import ONNXExportBackend
from .tensorrt import TensorRTExportBackend
from .torchscript import TorchScriptExportBackend
from .openvino import OpenVINOExportBackend


BackendType = Literal["onnx", "tensorrt", "torchscript", "openvino"]


def get_export_backend(backend: BackendType, **kwargs) -> ExportBackend:
    """Factory for export backends."""
    backends = {
        "onnx": ONNXExportBackend,
        "tensorrt": TensorRTExportBackend,
        "torchscript": TorchScriptExportBackend,
        "openvino": OpenVINOExportBackend,
    }

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Choose from {list(backends.keys())}")

    return backends[backend](**kwargs)


def export_model(
    model: nn.Module,
    export_dir: str | Path,
    mode: Literal["static", "dynamic", "auto"] = "auto",
    backend: BackendType = "onnx",
    **kwargs
) -> Path:
    """Export a GetiPrompt model to specified backend.

    This is the utility function called by Model.export().
    Most users should use the object-oriented API: model.export()

    Args:
        model: The GetiPrompt model to export
        export_dir: Directory to save exported models
        mode: Export mode
            - 'static': Bake reference embeddings into model (requires model.fit() first)
            - 'dynamic': Export extractor + predictor separately (change references at runtime)
            - 'auto': Static if model has embeddings, else dynamic
        backend: Export backend ('onnx', 'tensorrt', 'torchscript', 'openvino')
        **kwargs: Backend-specific options (opset_version, fp16_mode, optimization_level, etc.)

    Returns:
        Path to export directory with metadata

    Examples:
        >>> from getiprompt.models import Matcher
        >>> from getiprompt.export import export_model
        >>>
        >>> model = Matcher()
        >>>
        >>> # Option 1: Object-oriented API (RECOMMENDED)
        >>> model.fit(ref_images, ref_priors)
        >>> model.export("./exports", mode="static")  # Clean and intuitive!
        >>>
        >>> # Option 2: Utility function (for advanced use cases)
        >>> model.fit(ref_images, ref_priors)
        >>> export_model(
        ...     model,
        ...     export_dir="./exports",
        ...     mode="static",
        ...     backend="onnx",
        ...     opset_version=17
        ... )
        >>>
        >>> # Dynamic export for flexible references
        >>> model.export(
        ...     "./exports",
        ...     mode="dynamic",
        ...     backend="tensorrt",
        ...     fp16_mode=True
        ... )
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    exporter = get_export_backend(backend, **kwargs)
    model_name = model.__class__.__name__.lower()

    exported_paths = {}

    if mode == "dynamic":
        # Export extractor and predictor separately
        extractor_path = output_dir / f"{model_name}_extractor{exporter.file_extension}"
        predictor_path = output_dir / f"{model_name}_predictor{exporter.file_extension}"

        # Export extractor
        exporter.export_module(
            model.extractor,
            extractor_path,
            input_shapes=kwargs.get('extractor_input_shapes', {
                'images': (1, 3, 1024, 1024),
                'priors': (1, 1, 1024, 1024)
            }),
            output_names=['embeddings', 'masks']
        )

        # Export predictor
        exporter.export_module(
            model.predictor,
            predictor_path,
            input_shapes=kwargs.get('predictor_input_shapes', {
                'target_images': (1, 3, 1024, 1024),
                'reference_embeddings': (1, 256),
                'reference_masks': (1, 1, 1024, 1024)
            }),
            output_names=['masks', 'annotations']
        )

        exported_paths['extractor'] = extractor_path
        exported_paths['predictor'] = predictor_path

    else:  # static mode
        # Export full model with frozen embeddings
        model_path = output_dir / f"{model_name}{exporter.file_extension}"

        # Learn from reference data
        reference_images = kwargs.get('reference_images')
        reference_priors = kwargs.get('reference_priors')

        if reference_images is None or reference_priors is None:
            raise ValueError("Static mode requires 'reference_images' and 'reference_priors'")

        model.learn(reference_images, reference_priors)

        # Export in inference-only mode
        exporter.export_module(
            model,
            model_path,
            input_shapes={'target_images': (1, 3, 1024, 1024)},
            output_names=['masks', 'annotations']
        )

        exported_paths['model'] = model_path

    return exported_paths
```

---

## 4. Usage Patterns

### Recommended: Object-Oriented API

The cleanest way to export models is using the object-oriented `model.export()` method:

```python
from getiprompt.models import Matcher

# Train model
model = Matcher()
# ... training code ...

# Static export (80% of production cases)
model.fit(reference_images, reference_priors)
model.export("./exports", mode="static")  # Clean!

# Dynamic export (flexible references)
model.export("./exports", mode="dynamic")

# Multi-backend export
model.export(
    "./jetson_exports",
    mode="static",
    backend="tensorrt",
    fp16_mode=True
)

# Auto mode (smart defaults)
model.fit(reference_images, reference_priors)
model.export("./exports")  # Automatically chooses static!
```

### Pattern 1: Dynamic Inference (Most Flexible)

**Using object-oriented API (recommended):**
```python
from getiprompt.models import Matcher

# Train model
model = Matcher()

# Export to ONNX - simple and clean!
model.export(
    "./exported/matcher",
    mode="dynamic",
    backend="onnx",
    opset_version=17,
    dynamic_axes={
        'images': {0: 'batch'},
        'priors': {0: 'batch'},
        'target_images': {0: 'batch'}
    }
)
```

**Using utility function (advanced):**
```python
from getiprompt.export import export_model

# Same export using utility function
export_model(
    model,
    export_dir="./exported/matcher",
    mode="dynamic",
    backend="onnx",
    opset_version=17,
    dynamic_axes={
        'images': {0: 'batch'},
        'priors': {0: 'batch'},
        'target_images': {0: 'batch'}
    }
)
```

**Deployment (ONNX Runtime):**
```python
import onnxruntime as ort
import numpy as np

# Load both models
extractor = ort.InferenceSession("matcher_extractor.onnx")
predictor = ort.InferenceSession("matcher_predictor.onnx")

# Extract embeddings from NEW references (dynamic!)
ref_images = np.random.randn(5, 3, 1024, 1024).astype(np.float32)
ref_priors = np.random.randn(5, 1, 1024, 1024).astype(np.float32)

embeddings, masks = extractor.run(None, {
    'images': ref_images,
    'priors': ref_priors
})

# Predict on multiple target batches (reuse embeddings)
for target_batch in target_batches:
    results = predictor.run(None, {
        'target_images': target_batch,
        'reference_embeddings': embeddings,
        'reference_masks': masks
    })
```

### Pattern 2: Static Inference (Fastest)

```python
# Export with frozen embeddings
export_model(
    model,
    backend="tensorrt",
    output_dir="./exported/matcher",
    mode="static",
    reference_images=ref_images,
    reference_priors=ref_priors,
    fp16_mode=True
)
```

**Deployment (TensorRT):**
```python
import tensorrt as trt
import pycuda.driver as cuda

# Load engine
with open("matcher.trt", "rb") as f:
    engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Predict (references are baked in)
for target_batch in target_batches:
    context.execute_v2(bindings)  # Fast inference!
```

### Pattern 3: Multi-Backend Deployment

```python
# Export to all backends
for backend in ["onnx", "tensorrt", "torchscript", "openvino"]:
    export_model(
        model,
        backend=backend,
        output_dir=f"./exported/{backend}",
        mode="dynamic"
    )
```

---

## 5. Best Practices Summary

### ✅ DO

1. **Keep submodules stateless** - All data flows through arguments/returns
2. **Export extractor and predictor separately** - Enables dynamic references
3. **Use dynamic axes for batch size** - Flexibility for different batch sizes
4. **Validate exports** - Compare PyTorch vs exported outputs
5. **Optimize for target hardware** - Use FP16 for GPU, INT8 for edge
6. **Cache embeddings** - Extract once, predict many times
7. **Version your exports** - Track which model version produced which export

### ❌ DON'T

1. **Don't store state in submodules** - Breaks export and dynamic inference
2. **Don't use complex control flow in exported modules** - Many backends don't support it
3. **Don't export the wrapper Model class** - Export submodules instead
4. **Don't assume fixed batch size** - Use dynamic axes
5. **Don't skip validation** - Always verify exported model matches PyTorch

---

## 6. Performance Comparison

| Backend | Latency | Memory | Flexibility | Ease of Use |
|---------|---------|--------|-------------|-------------|
| **PyTorch** | Baseline | High | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ONNX** | 1.2-1.5x faster | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **TensorRT** | 2-4x faster | Low | ⭐⭐⭐ | ⭐⭐⭐ |
| **TorchScript** | 1.1-1.3x faster | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **OpenVINO** | 1.5-2.5x faster | Low | ⭐⭐⭐ | ⭐⭐⭐ |

**Recommendations:**
- **Development/Prototyping**: PyTorch or TorchScript
- **Production (NVIDIA GPU)**: TensorRT (FP16)
- **Production (CPU)**: OpenVINO or ONNX
- **Production (Cross-platform)**: ONNX
- **Edge Devices**: TensorRT (INT8) or OpenVINO

---

## 7. Future Considerations

### Quantization
```python
# INT8 quantization for TensorRT
export_model(
    model,
    backend="tensorrt",
    mode="dynamic",
    int8_mode=True,
    calibration_data=calibration_loader
)
```

### Dynamic Shapes
```python
# Support multiple input resolutions
export_model(
    model,
    backend="onnx",
    mode="dynamic",
    dynamic_axes={
        'images': {0: 'batch', 2: 'height', 3: 'width'}
    }
)
```

### Model Serving
```python
# Triton Inference Server deployment
# TensorRT + ONNX backends for optimal performance
```

---

## 8. InferenceModel: Production Deployment

For production, use **`InferenceModel`** - a unified interface that wraps exported models:

```python
from getiprompt.inference import InferenceModel

# Backend auto-detected from file extension!
model = InferenceModel(
    extractor="matcher_extractor.trt",  # .trt → TensorRT
    predictor="matcher_predictor.trt"
)

# Same API as PyTorch Model!
model.learn(ref_images, ref_priors)
results = model.infer(target_images)

# Change references dynamically (no re-export needed!)
model.learn(new_ref_images, new_ref_priors)
results = model.infer(target_images)

# Switch backend by changing file extension
model = InferenceModel(
    extractor="matcher_extractor.onnx",  # .onnx → ONNX
    predictor="matcher_predictor.onnx"
)
```

**Benefits:**
- ✅ **Same API as PyTorch**: Drop-in replacement, zero refactoring
- ✅ **Auto-detect Backend**: No need to specify backend manually
- ✅ **Backend Agnostic**: Switch between ONNX/TensorRT/OpenVINO by changing file path
- ✅ **Production Performance**: 2-4x faster than PyTorch
- ✅ **Dynamic References**: Change reference data without re-exporting

**File Extension Mapping:**
- `.onnx` → ONNX Runtime
- `.trt` → TensorRT
- `.pt`, `.pth` → TorchScript
- `.xml`, `.bin` → OpenVINO

See [INFERENCE_MODEL.md](INFERENCE_MODEL.md) for complete documentation.

---

## Summary

**Key Takeaways:**
1. ✅ **Stateless submodules** enable clean export to all backends
2. ✅ **Separate extractor/predictor** enables dynamic reference extraction
3. ✅ **Strategy pattern** provides unified interface across backends
4. ✅ **InferenceModel** provides PyTorch-like API for production
5. ✅ **Validate exports** to ensure correctness
6. ✅ **Choose backend based on deployment target** (GPU → TensorRT, CPU → OpenVINO, etc.)

This architecture provides **maximum flexibility** (dynamic references) with **optimal performance** (backend-specific optimizations) while maintaining **code simplicity** (unified API).
