# Backend-Agnostic Model Contract

---

## Problem

`Model` inherits from `nn.Module`. This forces every model — including OpenVINO-only ones — to import PyTorch. That makes OV-only deployments impossible without shipping the full torch wheel (~2 GB), and it adds unnecessary startup latency.

Current problem examples:

- `SAM3OpenVINO` inherits `nn.Module` just to satisfy the base class, then wraps numpy arrays in tensors at the boundary for no real reason.
- `predict()` returns `list[dict[str, torch.Tensor]]` — callers without torch can't use the output.
- The backend app has parallel `TorchModelHandler` / `OpenVINOModelHandler` classes duplicating lifecycle logic.

## Goals

1. Base `Model` contract imports zero torch.
2. OV models work in environments where torch is not installed.
3. Application backend calls `model.predict(batch)` directly — no model specific handler in application backend. Application should be able to have same API for models with different backend. 
4. Application backend should not have to do any numpy/torch related operations (like resizing). 


---

## Design

### Class hierarchy

```
                     Model(ABC)              ← torch-free
                    /          \
          TorchModel            OpenVINOModel
        (Model, nn.Module)      (Model)
           /     \                /       \
        SAM3    Matcher    SAM3OV   MatcherOV
```

Sibling models: each backend gets its own class. No runtime branching on backend inside a single class.

`TorchModel` and `OpenVINOModel` are intermediate bases that hold backend-specific boilerplate. Concrete models (SAM3, Matcher, etc.) inherit from one of these.

### `Model` — base contract

**Currently:** `Model(nn.Module)` in `library/src/instantlearn/models/base.py`. Accepts an optional `PostProcessor`, defines `fit()`, `predict()` (returning torch tensors), and `export()`.

**Problem:** Inheriting `nn.Module` means every model — including pure OV ones — needs torch installed. The predict signature returns torch tensors, making the output unusable without torch.

**Proposed:**

```python
# library/src/instantlearn/models/base.py

class Model(ABC):

    @classmethod
    @abstractmethod
    def card(cls) -> ModelCard: ...

    @property
    @abstractmethod
    def backend(self) -> Backend: ...

    @abstractmethod
    def fit(self, prompts: Sequence[Prompt]) -> None:
        """Load reference prompts. Idempotent — calling again replaces state."""

    @abstractmethod
    def predict(self, batch: Sequence[Sample]) -> list[Prediction]:
        """Run inference. Inputs/outputs are numpy."""
```

Pure ABC — no torch import at module level. `export()` moves down to `TorchModel` since it's a torch-specific concern.

### `TorchModel` — torch-aware intermediate

**Currently:** All torch models inherit `Model(nn.Module)` directly. There's no intermediate class — each model independently implements predict() returning `list[dict[str, torch.Tensor]]`.

**Problem:** No shared place to put torch-specific concerns (export, device movement, `inference_mode` context). Each model re-implements these patterns.

**Proposed:**

```python
# library/src/instantlearn/models/torch_base.py

class TorchModel(Model, nn.Module):

    @property
    def backend(self) -> Backend:
        return Backend.TORCH

    @abstractmethod
    def predict(self, batch: Sequence[Sample]) -> list[Prediction]:
        """Subclasses implement this directly with torch internals,
        but must return Prediction (numpy-based)."""

    @abstractmethod
    def export(self, path: Path) -> Path:
        """Export to OV IR. Returns .xml path."""

    @abstractmethod
    def to_openvino(self) -> Model:
        """Export + load the OV sibling. Each model implements its own conversion."""
        ...
```

Subclasses implement `predict()` directly — no indirection through a separate method. The conversion from internal torch tensors to the numpy-based `Prediction` output is the subclass's responsibility (just a `.cpu().numpy()` at the return boundary).

`to_openvino()` is abstract — each model knows its own export quirks (graph tracing, dynamic axes, which submodels to export separately).

### `OpenVINOModel` — OV-aware intermediate

**Currently:** No OV base class exists. `SAM3OpenVINO` inherits `Model(nn.Module)` directly, importing torch just to satisfy the base class.

**Problem:** Each OV model independently sets up `ov.Core()`, compiles, creates infer requests. No shared place for OV-specific patterns (device selection, model loading, IR compilation).

**Proposed:**

```python
# library/src/instantlearn/models/openvino_base.py

class OpenVINOModel(Model):
    """Base for all OpenVINO-backed models. No torch dependency."""

    def __init__(self, model_path: Path, device: str = "AUTO") -> None:
        self._core = ov.Core()
        self._compiled = self._core.compile_model(model_path, device)
        self._device = device

    @property
    def backend(self) -> Backend:
        return Backend.OPENVINO

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, cache_dir: str | None = None, **kwargs
    ) -> "OpenVINOModel":
        """Load from HF repo or local path. HF-style signature."""
        ...
```

Concrete OV models inherit from this:

```python
# library/src/instantlearn/models/sam3/sam3_openvino.py

class SAM3OpenVINO(OpenVINOModel):

    def __init__(self, model_path: Path, device: str = "AUTO") -> None:
        super().__init__(model_path, device)
        self._request = self._compiled.create_infer_request()
        ...

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, cache_dir: str | None = None, **kwargs
    ) -> "SAM3OpenVINO":
        # most models won't load from HF due to license, but useful for future models
        ...

    @classmethod
    def card(cls) -> ModelCard:
        return SAM3.card()  # delegates to torch sibling

    def fit(self, prompts: Sequence[Prompt]) -> None: ...
    def predict(self, batch: Sequence[Sample]) -> list[Prediction]: ...
```

### Construction patterns

```python
# Torch
model = SAM3()
model.fit(prompts)
preds = model.predict(images)

# OV from hub/disk
model = SAM3OpenVINO.from_pretrained("intel/sam3-ov-int8")

# Torch → OV
# Backend can just call to_openvino() if needed
ov_model = SAM3().to_openvino()
```


---

## `Prediction` dataclass
Currently we use a dict for results. This is a bit non-standard. I suggest a new Prediction class that's based on numpy. 
However, we might need torch tensor output sometimes - maybe another model consumes the output or for calculuating metrics using torchmetrics. For this, we can have a to_tensors() or to_torch() method which lazy imports torch. 

```python
@dataclass(frozen=True)
class Prediction:
    masks: np.ndarray               # (N, H, W) bool/uint8
    scores: np.ndarray              # (N,) float32
    label_ids: np.ndarray           # (N,) int32
    label_names: np.ndarray         # (N,) str/object
    boxes: np.ndarray | None = None # (N, 4) xyxy float32
    points: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def to_tensors(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Lazy torch import, zero-copy where possible. For torchmetrics."""
        import torch
        out = {"masks": torch.from_numpy(self.masks).to(device), ...}
        return out
```

Frozen dataclass — thread-safe, cacheable, forward-compatible.

---

## Processors
Introduce preprocessor that is similar to postprocessor. This should help in removing all the resizing and other such operations from the backend side. 
Pre- and post-processors drop `nn.Module` and become plain ABCs over numpy:

```python
class Preprocessor(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray: ...

class PreprocessorPipeline(Preprocessor):
    def __init__(self, steps: list[Preprocessor]) -> None:
        self._steps = steps

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for step in self._steps:
            image = step(image)
        return image
```

Same pattern for `PostProcessor` / `PostProcessorPipeline` (operates on masks, scores, labels).

No `|` operator for chaining — `Pipeline([a, b, c])` reads better for CV folks and is easier to grep.

Static postprocessing that needs to be baked into an exported model stays inside each model's `export()` method.

---

## Application layer changes

Delete `TorchModelHandler` and `OpenVINOModelHandler`. Factory returns `Model` directly:

```python
def create_model(config: ModelConfig, backend: Backend) -> Model:
    cls = _resolve_class(config, backend)
    if backend is Backend.OPENVINO:
        return cls.from_pretrained(config.repo_id)
    return cls(**config.kwargs)
```

Inference becomes standard irrespective of the model backend. 

```python
predictions = model.predict(batch)
```

---

## Migration plan
Not sure of this one. We can discuss this. 
One model per PR might be too slow , so we can YOLO everything :) 


