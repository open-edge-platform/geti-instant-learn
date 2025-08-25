from abc import ABC, abstractmethod
from typing import Dict, Any

from backend.app.pipeline.core.base import StreamReader, StreamWriter
from types import ConfigDict


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
