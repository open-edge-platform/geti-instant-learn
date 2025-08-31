#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Any

from backend.app.runtime.core.base import StreamWriter
from backend.app.runtime.core.types import ConfigDict


class NoOpWriter(StreamWriter):

    def get_config(self) -> ConfigDict:
        return {}

    def write(self, data: Any) -> None:
        pass
