from runtime.core.components.base import StreamWriter
from runtime.core.components.schemas.writer import WriterConfig


class StreamWriterFactory:

    @classmethod
    def create(cls, config: WriterConfig) -> StreamWriter:
        pass
