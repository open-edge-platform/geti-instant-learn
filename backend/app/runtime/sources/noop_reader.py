#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Any

from backend.app.runtime.core.base import StreamReader


class NoOpReader(StreamReader):

    def read(self) -> Any | None:
        return None
