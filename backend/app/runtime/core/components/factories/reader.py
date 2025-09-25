from runtime.core.components.schemas.reader import ReaderConfig
from runtime.core.components.base import StreamReader


class StreamReaderFactory:

    @classmethod
    def create(cls, config: ReaderConfig) -> StreamReader:
        pass
