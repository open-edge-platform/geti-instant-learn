#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

import numpy as np

# todo: to be defined
ProcessorConfig = Any


@dataclass(kw_only=True)
class InputData:
    frame: np.ndarray  # frame loaded as numpy array
    context: dict[str, Any]  # unstructured metadata about the source of the frame (camera ID, video file, etc.)


@dataclass(kw_only=True)
class OutputData:
    frame: np.ndarray  # frame loaded as numpy array
    # the rest will be defined later.
