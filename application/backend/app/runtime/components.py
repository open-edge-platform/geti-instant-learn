#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from domain.db.models import PromptType
from domain.services.label import LabelService
from domain.services.project import ProjectService
from domain.services.prompt import PromptService
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
    def create_source(self, project_id: UUID) -> Source: ...

    @abstractmethod
    def create_processor(self, project_id: UUID) -> Processor: ...

    @abstractmethod
    def create_sink(self, project_id: UUID) -> Sink: ...


class DefaultComponentFactory(ComponentFactory):
    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def create_source(self, project_id: UUID) -> Source:
        with self._session_factory() as session:
            svc = ProjectService(session=session)
            cfg = svc.get_pipeline_config(project_id)
        return Source(StreamReaderFactory.create(cfg.reader))

    def create_processor(self, project_id: UUID) -> Processor:
        with self._session_factory() as session:
            prompt_svc = PromptService(session)
            project_svc = ProjectService(session)
            label_svc = LabelService(session)
            cfg = project_svc.get_pipeline_config(project_id)
            label_colors = label_svc.get_label_colors_for_processor(project_id)

            reference_batch = None
            category_id_to_label_id: dict[int, str] = {}

            batch_result = prompt_svc.get_reference_batch(project_id, PromptType.VISUAL)
            if batch_result is not None:
                reference_batch, category_id_to_label_id = batch_result

            logger.info("Creating processor with model config: %s", cfg.processor)
            logger.debug(
                "Category mapping: %s, Label colors: %s",
                category_id_to_label_id,
                list(label_colors.keys()),
            )

        return Processor(
            model_handler=ModelFactory.create(reference_batch, cfg.processor),
            batch_size=get_settings().processor_batch_size,
            label_colors=label_colors,
            category_id_to_label_id=category_id_to_label_id,
        )

    def create_sink(self, project_id: UUID) -> Sink:
        with self._session_factory() as session:
            svc = ProjectService(session=session)
            cfg = svc.get_pipeline_config(project_id)
        return Sink(StreamWriterFactory.create(cfg.writer))
