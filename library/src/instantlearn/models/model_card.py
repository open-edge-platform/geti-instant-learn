# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ModelCard dataclass describing model capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instantlearn.utils.constants import Backend, PromptType, ShotMode


@dataclass(frozen=True)
class ModelCard:
    """Immutable descriptor of a model's capabilities.

    Frozen — instances can be used as dict keys or set members. OV siblings
    delegate their ``card()`` classmethod to the torch sibling so the card
    describes what the model *can do*, independent of the current runtime
    backend.

    Attributes:
        name: Human-readable model name, e.g. ``"SAM3"``.
        family: Groups torch and OV siblings under one identifier,
            e.g. ``"sam3"``.
        description: One-liner shown in tooltips and logs.
        prompt_types: Set of prompt types the model accepts.
        shot_modes: Set of shot modes the model supports.
        exportable_to: Backends this model can be exported to.

    Example:
        >>> ModelCard(
        ...     name="SAM3",
        ...     family="sam3",
        ...     description="Segment Anything 3 — text and visual prompting",
        ...     prompt_types=frozenset({PromptType.TEXT, PromptType.MASK}),
        ...     shot_modes=frozenset({ShotMode.ZERO_SHOT, ShotMode.ONE_SHOT}),
        ...     exportable_to=frozenset({Backend.OPENVINO, Backend.ONNX}),
        ... )
    """

    name: str
    family: str
    description: str
    prompt_types: frozenset[PromptType]
    shot_modes: frozenset[ShotMode]
    exportable_to: frozenset[Backend]
