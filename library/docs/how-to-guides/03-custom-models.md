# How to Create a Custom Model

Any class implementing the `Model` contract works with the rest of the
library — the CLI, the benchmark harness and the application backend all talk to
it through the same four members.

## Choose a base class

| Base            | Use when                                        |
| --------------- | ----------------------------------------------- |
| `TorchModel`    | Your model runs PyTorch                         |
| `OpenVINOModel` | Your model loads an already-exported OpenVINO IR |
| `Model`         | Any other runtime (ONNX Runtime, TensorRT, ...) |

Inherit from `Model` directly only if neither backend base fits; it is an ABC
with no runtime behaviour of its own.

## The contract

Four members, three of them abstract:

```python
from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Sample
from instantlearn.models.model_card import ModelCard
from instantlearn.models.torch_base import TorchModel


class MyModel(TorchModel):
    @classmethod
    def card(cls) -> ModelCard:
        """Static capabilities — callable without instantiating."""

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Accept the prompt."""

    def predict(self, target: Sample | list[Sample] | Batch) -> list[Prediction]:
        """One Prediction per input sample, in order."""
```

`backend` is supplied by `TorchModel` and `OpenVINOModel`, so you only
implement it when inheriting `Model` directly.

## Write the card

Declare the card once at module level so it is cheap to read and impossible to
get out of sync between siblings:

```python
# _card.py
from instantlearn.models.model_card import ModelCard
from instantlearn.utils.constants import Backend, PromptType, ShotMode

_MY_MODEL_CARD = ModelCard(
    name="MyModel",
    family="my_model",
    description="One-line summary shown in tooltips and logs.",
    prompt_types=frozenset({PromptType.MASK, PromptType.TEXT}),
    shot_modes=frozenset({ShotMode.ONE_SHOT, ShotMode.FEW_SHOT}),
    exportable_to=frozenset({Backend.OPENVINO}),
)
```

An OpenVINO sibling reuses the torch model's card, since a card describes
capability rather than runtime:

```python
class MyModelOpenVINO(OpenVINOModel):
    @classmethod
    def card(cls) -> ModelCard:
        return _MY_MODEL_CARD
```

## Implement fit()

`fit()` stores whatever `predict()` will need. It must be idempotent — calling
it again replaces the previous prompt:

```python
from instantlearn.models.torch_adapter import CategoryRegistry, samples_to_tensors


def fit(self, reference: Sample | list[Sample] | Batch) -> None:
    samples = samples_to_tensors(reference, self.device)
    self.categories = CategoryRegistry.from_samples(samples)
    self.reference_features = self._encode(samples)
```

`samples_to_tensors()` is the single numpy-to-torch boundary. It accepts all
three input shapes, so you never branch on the input type yourself.

`CategoryRegistry` holds the id-to-name mapping so predictions can report
`label_names`. Build it in `fit()` and keep it.

## Implement predict()

Convert at the boundary, infer natively, convert back:

```python
from instantlearn.models.torch_adapter import (
    prediction_categories_for_sample,
    tensors_to_prediction,
)
from instantlearn.utils.errors import ModelNotFittedError


def predict(self, target: Sample | list[Sample] | Batch) -> list[Prediction]:
    if self.reference_features is None:
        msg = "predict() requires fit() to be called first."
        raise ModelNotFittedError(msg)

    predictions = []
    for sample in samples_to_tensors(target, self.device):
        masks, scores, labels = self._infer(sample)
        predictions.append(
            tensors_to_prediction(
                masks=masks,
                scores=scores,
                label_ids=labels,
                categories=prediction_categories_for_sample(self.categories, sample),
            ),
        )
    return predictions
```

Three rules that keep a model interchangeable with the built-ins:

1. **Return one `Prediction` per input sample, in order.** Callers zip results
   against inputs.
2. **Raise `ModelNotFittedError`**, not a bare `RuntimeError`, when a prompt is
   required but missing.
3. **Emit masks at the input frame's resolution.** Rescale internally; do not
   make the caller do it.

`tensors_to_prediction()` enforces the contract dtypes (`bool`/`uint8` masks,
`float32` scores, `int32` label ids), so use it rather than building a
`Prediction` by hand.

## Support export (optional)

`TorchModel` declares `to_openvino()` abstract, so a torch model must provide
it. If export makes no sense for your model, raise:

```python
def to_openvino(self, export_path=None, config=None):
    msg = "MyModel does not support OpenVINO export."
    raise NotImplementedError(msg)
```

Otherwise trace a graph module and convert it. Freeze anything learned in
`fit()` as buffers so it is captured during tracing — that is why models like
`Matcher` require `fit()` before `to_openvino()`.

## Register the model

Export it from the package so it is importable from `instantlearn.models`:

```python
# instantlearn/models/my_model/__init__.py
from .my_model import MyModel

__all__ = ["MyModel"]
```

```python
# instantlearn/models/__init__.py
from .my_model import MyModel
```

## Check your work

```python
from instantlearn.models import Model

assert isinstance(model, Model)
assert model.card().name
assert model.backend

predictions = model.predict([sample_a, sample_b])
assert len(predictions) == 2
assert predictions[0].masks.shape[1:] == sample_a.image.shape[:2]
```

## Next Steps

- [Architecture](../concepts/02-architecture.md) — where your model fits
- [Core Concepts](../concepts/01-concepts.md) — `Sample` and `Prediction`
