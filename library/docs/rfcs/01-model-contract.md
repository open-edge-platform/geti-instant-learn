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
3. Torch models retain full tensor ergonomics for training and metrics.
4. App layer calls `model.predict(batch)` directly — no handler indirection.


---

## Design

### Class hierarchy

```
                     Model(ABC)          ← torch-free
                    /          \
          TorchModel            OVModels
        (Model, nn.Module)      (SAM3OpenVINO, MatcherOV, ...)
           /     \
        SAM3    Matcher ...
```

Sibling models: each backend gets its own class. No runtime branching on backend inside a single class.

### `Model` — base contract

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

### `TorchModel` — torch-aware intermediate

```python
# library/src/instantlearn/models/torch_base.py

class TorchModel(Model, nn.Module):

    @property
    def backend(self) -> Backend:
        return Backend.TORCH

    @final
    def predict(self, batch: Sequence[Sample]) -> list[Prediction]:
        tensors = self._to_tensors(batch)
        with torch.inference_mode():
            outputs = self.predict_tensors(tensors)
        return self._to_predictions(outputs)

    @abstractmethod
    def predict_tensors(self, batch: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Torch-native path — use for training loops, custom metrics."""

    @abstractmethod
    def export(self, path: Path) -> Path:
        """Export to OV IR. Returns .xml path."""

    def to_openvino(self) -> Model:
        """Export + load the OV sibling."""
        ...
```

`predict()` is `@final` — subclasses implement `predict_tensors()` only. The base handles tensor↔numpy conversion.

### OpenVINO siblings

```python
# library/src/instantlearn/models/sam3/sam3_openvino.py

class SAM3OpenVINO(Model):

    def __init__(self, model_path: Path, device: str = "AUTO") -> None:
        self._core = ov.Core()
        self._compiled = self._core.compile_model(model_path, device)
        ...

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs) -> "SAM3OpenVINO": ...

    @classmethod
    def card(cls) -> ModelCard:
        return SAM3.card()  # delegates to torch sibling

    @property
    def backend(self) -> Backend:
        return Backend.OPENVINO

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
ov_model = SAM3().to_openvino()
```

No `from_torch()` — use `to_openvino()` on the torch instance instead.

---

## `Prediction` dataclass

```python
@dataclass(frozen=True)
class Prediction:
    masks: np.ndarray               # (N, H, W) bool/uint8
    scores: np.ndarray              # (N,) float32
    labels: np.ndarray              # (N,) int32
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

Inference becomes:

```python
predictions = model.predict(batch)
```

---

## Migration plan

One model per PR, mechanical:

1. Land base types (`Model`, `TorchModel`, `Prediction`, processor ABCs). Keep old `Model` as deprecated alias for one release.
2. Migrate each torch model: inherit `TorchModel`, rename `predict` → `predict_tensors`, add `card()`.
3. Migrate OV models: drop `nn.Module`, return `Prediction` from `predict()`.
4. Rewrite postprocessors to numpy ABCs.
5. Delete handler classes in the backend app; update factory.
6. Drop deprecated alias.

**Acceptance criterion:** `pip install instantlearn[openvino]` (no torch extra) works end-to-end with `SAM3OpenVINO`.

---

## Deferred

| Topic | Reason | Tracked in |
|-------|--------|------------|
| Device pool / management | Separate concern, needs its own discussion | TBD |
| fit() state machine | No bug motivating it yet | — |
| Async predict | App-level concern | — |

---


