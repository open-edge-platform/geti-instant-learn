from runtime.core.components.base import Processor
from runtime.core.components.schemas.processor import ProcessorConfig, InputData, OutputData


class ProcessorFactory:

    @classmethod
    def create(cls, config: ProcessorConfig) -> Processor[InputData, OutputData]:
        pass
