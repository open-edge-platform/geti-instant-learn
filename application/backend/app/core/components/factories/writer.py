#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from core.components.base import StreamWriter
from core.components.schemas.writer import WriterConfig


class StreamWriterFactory:
    """
    A factory for creating StreamWriter instances based on a configuration.

    This class decouples the application from the concrete implementation of
    the StreamWriter, allowing for different writer types to be instantiated
    based on the provided configuration.
    """

    @classmethod
    def create(cls, config: WriterConfig) -> StreamWriter:
        pass
