#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
from pydantic import BaseModel


# Generic type variables for pipeline data flow

class InputData(BaseModel):
    frame: np.ndarray  # frame loaded as numpy array
    context: dict[str, Any]  # unstructured metadata about the source of the frame (camera ID, video file, etc.)


class OutputData(BaseModel):
    frame: np.ndarray  # frame loaded as numpy array
    # the rest will be defined later.
