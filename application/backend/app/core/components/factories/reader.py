#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from core.components.base import StreamReader
from core.components.schemas.reader import ReaderConfig


class StreamReaderFactory:
    """
    A factory for creating StreamReader instances based on a configuration.

    This class decouples the application from the concrete implementation of
    the StreamReader, allowing for different reader types to be instantiated
    based on the provided configuration.
    """

    @classmethod
    def create(cls, config: ReaderConfig) -> StreamReader:
        pass
