#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Any

from backend.app.runtime.core.base import StreamWriter


class NoOpWriter(StreamWriter):

    def write(self, data: Any) -> None:
        pass
