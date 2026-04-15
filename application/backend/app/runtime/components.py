#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod

from instantlearn.data.base.batch import Batch
from sqlalchemy.orm import Session, sessionmaker

from domain.services.project import ProjectService
from domain.services.schemas.device import AvailableDeviceSchema
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.reader import ReaderConfig
from domain.services.schemas.writer import WriterConfig
from runtime.core.components.factories.model import ModelFactory
from runtime.core.components.factories.reader import StreamReaderFactory
from runtime.core.components.factories.writer import StreamWriterFactory
from runtime.core.components.processor import Processor
from runtime.core.components.sink import Sink
from runtime.core.components.source import Source
from settings import get_settings

logger = logging.getLogger(__name__)


class ComponentFactory(ABC):
    @abstractmethod
    def create_source(self, reader_cfg: ReaderConfig) -> Source: ...

    @abstractmethod
    def create_processor(self, pipeline_cfg: PipelineConfig, reference_batch: Batch) -> Processor: ...

    @abstractmethod
    def create_sink(self, writer_cfg: WriterConfig) -> Sink: ...


class DefaultComponentFactory(ComponentFactory):
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        available_devices: list[AvailableDeviceSchema] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._model_factory = ModelFactory(available_devices=available_devices)

    def create_source(self, reader_cfg: ReaderConfig) -> Source:
        return Source(StreamReaderFactory.create(reader_cfg))

    def create_processor(self, pipeline_cfg: PipelineConfig, reference_batch: Batch) -> Processor:
        logger.info("Creating processor with model config: %s", pipeline_cfg.processor)
        settings = get_settings()
        return Processor(
            model_handler=self._model_factory.create(
                reference_batch=reference_batch,
                config=pipeline_cfg.processor,
                configured_device=pipeline_cfg.device,
            ),
            batch_size=settings.processor_batch_size,
            frame_skip_interval=settings.processor_frame_skip_interval,
            frame_skip_amount=settings.processor_frame_skip_amount,
        )

    def create_sink(self, writer_cfg: WriterConfig) -> Sink:
        return Sink(StreamWriterFactory.create(writer_cfg))
