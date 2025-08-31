#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Any

from backend.app.runtime.core.base import StreamReader
from backend.app.runtime.core.types import ConfigDict


class NoOpReader(StreamReader):

    def get_config(self) -> ConfigDict:
        return {}

    def read(self) -> Any | None:
        return None
