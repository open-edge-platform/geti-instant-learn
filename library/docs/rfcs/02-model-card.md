# ModelCard & Capability Enums

| Field | Value |
|-------|-------|
| Status | Draft |
| Parent | model-contract.md |
| Tracking issue | #1001 |

---

## Context

The `Model` base class exposes a `card()` classmethod returning a `ModelCard`. This document defines that type and the enums it uses.

The guiding principle: **only add fields that drive branching logic** in the app or UI. Display-only metadata (param counts, FLOPs, paper links) can come later if a consumer actually needs to query on them.

---

## Enums

```python
from enum import StrEnum

class PromptType(StrEnum):
    TEXT = "text"
    VISUAL = "visual"
    POINT = "point"
    BOX = "box"

class ShotMode(StrEnum):
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    FEW_SHOT = "few_shot"
```

These are `StrEnum` so they serialize cleanly to JSON for the frontend.

## `ModelCard`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelCard:
    name: str                           # e.g. "SAM3"
    family: str                         # groups siblings, e.g. "sam3"
    description: str                    # one-liner for tooltips
    prompt_types: frozenset[PromptType]
    shot_modes: frozenset[ShotMode]
    exportable_to: frozenset[Backend]
```

Six fields. Frozen — can be used as dict keys if needed.

### Usage

```python
class SAM3(TorchModel):
    @classmethod
    def card(cls) -> ModelCard:
        return ModelCard(
            name="SAM3",
            family="sam3",
            description="Segment Anything 3 — text and visual prompting",
            prompt_types=frozenset({PromptType.TEXT, PromptType.VISUAL, PromptType.POINT, PromptType.BOX}),
            shot_modes=frozenset({ShotMode.ZERO_SHOT, ShotMode.ONE_SHOT, ShotMode.FEW_SHOT}),
            exportable_to=frozenset({Backend.OPENVINO, Backend.ONNX}),
        )
```

OV siblings delegate: `SAM3OpenVINO.card()` returns `SAM3.card()`. The card describes what the model *can do* — the instance's `backend` property says where it's currently running.

### How this scales

With 5–20 models this is fine. If we grow to 50+ and need richer queries, we can add a registry or extend the card. But we don't add fields speculatively — only when a real consumer needs to branch on them.

For reference, here's how other libraries handle this:

- **HuggingFace**: per-model `PretrainedConfig` + `AutoModelForX` registrations — works at 1000+ models but heavy boilerplate.
- **timm**: flat `pretrained_cfg` dicts with naming conventions — simple but not queryable.
- **Ultralytics YOLO**: single `task` attribute — minimal.

We're closer to YOLO's simplicity right now, but with actual type safety.

---

## Not included

- `metadata()` classmethod — no consumer needs it today.
- Nested config objects — overkill for 6 fields.
- Versioning of the card schema — handle when we actually need to change it.
