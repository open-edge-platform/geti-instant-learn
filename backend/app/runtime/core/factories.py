#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from typing import Optional

from backend.app.runtime.core.base import StreamReader, StreamWriter, Processor
from .types import ConfigDict, IN, OUT


class StreamReaderFactory:
    """Abstract factory for creating StreamReader instances from a configuration."""

    @classmethod
    def create(cls, config: Optional[ConfigDict]) -> StreamReader:
        pass


class StreamWriterFactory:
    """Abstract factory for creating StreamWriter instances from a configuration."""

    @classmethod
    def create(cls, config: Optional[ConfigDict]) -> StreamWriter:
        pass


class ProcessorFactory:
    """Abstract factory for creating Processor instances from a configuration."""

    @classmethod
    def create(cls, config: Optional[ConfigDict]) -> Processor[IN, OUT]:
        pass
