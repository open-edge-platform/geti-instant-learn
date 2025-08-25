#  Copyright (C) 2022-2025 Intel Corporation
#  LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE


from backend.app.pipeline.core.base import StreamReader, StreamWriter

from .types import ConfigDict


class StreamReaderFactory:
    """Abstract factory for creating StreamReader instances from a configuration."""

    @classmethod
    def create(cls, config: ConfigDict) -> StreamReader:
        pass


class StreamWriterFactory:
    """Abstract factory for creating StreamWriter instances from a configuration."""

    @classmethod
    def create(cls, config: ConfigDict) -> StreamWriter:
        pass
