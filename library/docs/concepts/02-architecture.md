# Architecture

The library separates *what a model does* from *what runtime it runs on*. This
page describes that split and how data flows through it.

## Overview

Models form a three-level hierarchy:

```
Model (ABC)                 backend-neutral contract, imports no torch
├── TorchModel              PyTorch-backed  (nn.Module, Model)
└── OpenVINOModel           OpenVINO-backed (Model)
```

`Model` defines the whole public surface — `card()`, `backend`, `fit()` and
`predict()` — in terms of numpy `Sample` and `Prediction`. Because it never
mentions tensors, the same contract describes a PyTorch model and an OpenVINO
one without either leaking into the other.

The two intermediate bases carry only backend concerns:

| Base            | Adds                                                        |
| --------------- | ----------------------------------------------------------- |
| `TorchModel`    | `nn.Module` inheritance, `device`/`precision`, `to_openvino()` |
| `OpenVINOModel` | `ov.Core` setup, `model_dir` resolution, IR loading          |

Export lives on `TorchModel` alone, which is the natural place for it: only a
torch model has a graph to trace. An OpenVINO model loads an IR that already
exists, so it has nothing to export and is not asked to implement it.

## Torch and OpenVINO siblings

Several models ship as a pair — `SAM3`/`SAM3OpenVINO`, `Matcher`/`MatcherOpenVINO`,
`PerDino`/`PerDinoOpenVINO`, `SoftMatcher`/`SoftMatcherOpenVINO`.

The pair describes one model with two runtimes, so both siblings return the
same `ModelCard`. Each model package defines it once, in a dependency-free
`_card.py`, and both classes import it:

```python
# sam3/_card.py
_SAM3_CARD = ModelCard(name="SAM3", family="sam3", ...)

# sam3/sam3.py and sam3/sam3_openvino.py
@classmethod
def card(cls) -> ModelCard:
    return _SAM3_CARD
```

`card()` describes what the model *can do*; `backend` reports what it is
*currently running on*. Keeping those separate is why a card can be shared
between siblings while `backend` still differs. Importing the constant rather
than delegating to the torch class also means the OpenVINO sibling never needs
to import torch.

Producing the OpenVINO sibling's input is a one-off, local step:

```python
SAM3(device="cpu").to_openvino(export_path="./sam3-openvino")
```

Some models bake the reference into the exported graph, so they require `fit()`
before `to_openvino()`. `Matcher` and `PerDino` work this way — their exported
IR has the reference features frozen in as constants.

## ModelCard

`ModelCard` is a frozen dataclass describing capabilities without instantiating
anything, so a caller can pick a model before paying to load one:

| Field           | Meaning                                    |
| --------------- | ------------------------------------------ |
| `name`          | Human-readable name, e.g. `"SAM3"`         |
| `family`        | Groups siblings, e.g. `"sam3"`             |
| `description`   | One-liner for tooltips and logs            |
| `prompt_types`  | `TEXT`, `MASK`, `BOUNDING_BOX`, `POINT`    |
| `shot_modes`    | `ZERO_SHOT`, `ONE_SHOT`, `FEW_SHOT`        |
| `exportable_to` | Backends the model can be exported to      |

```python
from instantlearn.models import SAM3
from instantlearn.utils.constants import PromptType

if PromptType.TEXT in SAM3.card().prompt_types:
    ...  # safe to prompt with a category name
```

## Data flow

Both backends look identical from outside and differ only in the middle:

```
Sample (numpy, HWC)
      │
      ├── TorchModel ──── numpy -> tensors ── inference ── tensors -> numpy ──┐
      │                        (torch_adapter)                                │
      │                                                                       ├── Prediction (numpy)
      └── OpenVINOModel ── numpy ────────── inference ─────────── numpy ──────┘
```

Torch models convert at their own boundary, in one place
(`instantlearn.models.torch_adapter`), rather than on `Sample` or `Prediction`.
That inversion is deliberate: the neutral data classes never import torch, so
adding a backend does not require touching them.

OpenVINO models have no conversion step — OpenVINO already speaks numpy — which
is why their path is a straight line.

## Components

Reusable pieces live outside the models and are shared between them:

| Package                          | Contains                                     |
| -------------------------------- | -------------------------------------------- |
| `components/encoders`            | Image encoders (HuggingFace, timm)           |
| `components/feature_extractors`  | Masked feature extraction, reference features |
| `components/sam`                 | SAM decoder and predictor                    |
| `components/postprocessing`      | Composable NMS, filtering, morphology        |
| `data/torch`                     | Dataset readers (COCO, LVIS, PerSeg, folder) |

Post-processors are chainable and attach to any model via the `postprocessor`
argument. They are `nn.Module`s so the ONNX exporter can trace them into the
exported graph, which is how an exported IR keeps the same suppression
behaviour as the torch model.

## Next Steps

- [Core Concepts](01-concepts.md) — `Sample`, `Prediction` and the model API
- [Custom Models](../how-to-guides/03-custom-models.md) — implement your own
