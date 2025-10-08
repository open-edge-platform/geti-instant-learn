#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from core.components.base import StreamReader
from core.components.schemas.processor import InputData


class NoOpReader(StreamReader):
    """
    A 'no-operation' implementation of the StreamReader interface.

    This reader serves as a placeholder when an input source is required by the
    pipeline but has not been configured. It consistently returns None.
    """

    def read(self) -> InputData | None:
        return None
