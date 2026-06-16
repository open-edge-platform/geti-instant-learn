# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ModelCard dataclass describing model capabilities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instantlearn.utils.constants import Backend, PromptType, ShotMode


@dataclass(frozen=True)
class ModelCard:
    """Immutable descriptor for a model's capabilities.

    Only fields that drive branching logic in the app or UI are included.

    Attributes:
        name: Human-readable model name, e.g. `"SAM3"`.
        family: Groups sibling torch/OV variants, e.g. `"sam3"`.
        description: One-liner shown in tooltips / logs.
        prompt_types: Set of prompt types the model accepts.
        shot_modes: Set of shot modes the model supports.
        exportable_to: Backends this model can be exported to.
    """

    name: str
    family: str
    description: str
    prompt_types: frozenset[PromptType]
    shot_modes: frozenset[ShotMode]
    exportable_to: frozenset[Backend]
