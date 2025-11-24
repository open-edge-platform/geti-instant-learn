#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.writer import WriterConfig
from runtime.core.components.base import StreamWriter
from runtime.core.components.writers.noop_writer import NoOpWriter


class StreamWriterFactory:
    """
    A factory for creating StreamWriter instances based on a configuration.

    This class decouples the application from the concrete implementation of
    the StreamWriter, allowing for different writer types to be instantiated
    based on the provided configuration.
    """

    @classmethod
    def create(cls, config: WriterConfig) -> StreamWriter:
        match config:
            case _:
                return NoOpWriter()
