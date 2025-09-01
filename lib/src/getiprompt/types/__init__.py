# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Import all type classes
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

# Export all classes
__all__ = [
    "Annotations",
    "Data",
    "Features",
    "Image",
    "Masks",
    "Points",
    "Priors",
    "Prompt",
    "Similarities",
    "Results",
    "Text",
    "Boxes",
]
