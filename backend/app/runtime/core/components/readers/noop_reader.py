#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Any

from runtime.core.components.base import StreamReader


class NoOpReader(StreamReader):

    def read(self) -> Any | None:
        return None
