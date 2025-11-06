#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from typing import Any

from runtime.core.components.base import StreamWriter


class NoOpWriter(StreamWriter):
    """

    A 'no-operation' implementation of the StreamWriter interface.

    This writer serves as a placeholder when an output sink is required by the
    pipeline but has not been configured. It accepts data and effectively writes it to the void.

    """

    def write(self, data: Any) -> None:
        pass
