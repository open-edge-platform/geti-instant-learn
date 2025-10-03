#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from typing import Any

from core.components.base import StreamReader


class NoOpReader(StreamReader):
    """
    A 'no-operation' implementation of the StreamReader interface.

    This reader serves as a placeholder when an input source is required by the
    pipeline but has not been configured. It consistently returns None.
    """

    def read(self) -> Any | None:
        return None
