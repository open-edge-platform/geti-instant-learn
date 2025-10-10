# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for GetiPrompt."""

from getiprompt.types.annotations import Annotations
from getiprompt.types.boxes import Boxes
from getiprompt.types.data import Data
from getiprompt.types.features import Features
from getiprompt.types.image import Image
from getiprompt.types.masks import Masks
from getiprompt.types.points import Points
from getiprompt.types.priors import Priors, Prompt
from getiprompt.types.results import Results
from getiprompt.types.similarities import Similarities
from getiprompt.types.text import Text

__all__ = [
    "Annotations",
    "Boxes",
    "Data",
    "Features",
    "Image",
    "Masks",
    "Points",
    "Priors",
    "Prompt",
    "Results",
    "Similarities",
    "Text",
]
