#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager
from typing import Callable
from uuid import UUID
from sqlalchemy.orm import Session, sessionmaker
from core.components.schemas.processor import ModelConfig
from core.components.schemas.reader import (
    SourceType,
    WebCamConfig,
    ReaderConfig,
)
from core.components.schemas.writer import WriterConfig
from core.runtime.dispatcher import (
    ComponentConfigChangeEvent,
    ConfigChangeDispatcher,
    ConfigChangeEvent,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from core.runtime.pipeline import Pipeline
from core.runtime.schemas.pipeline import PipelineConfig
from services.project import ProjectService
from services.errors import ResourceNotFoundError
from services.schemas.project import ProjectRuntimeConfig




logger = logging.getLogger(__name__)




class PipelineManager:
    """
    Glues the app configuration and runtime layers together by managing the active Pipeline.

    This class listens for configuration change events and translates them into
    lifecycle actions for the running Job instance, such as starting, stopping,
    or performing a live configuration update.
    """

    def __init__(self, event_dispatcher: ConfigChangeDispatcher, session_factory: sessionmaker[Session]):
        self._event_dispatcher = event_dispatcher
        self._session_factory = session_factory
        self._pipeline: Pipeline | None = None



    @contextmanager
    def _project_service(self):
        """
        Context manager yielding a short‑lived ProjectService.
        Ensures session cleanup without callback indirection.
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
            yield svc
    
    
    def start(self) -> None:
        """
        Start pipeline for active project if present; subscribe to config events.
        """
        cfg = self._get_active_pipeline_config()
        if cfg:
            self._pipeline = Pipeline(pipeline_conf=cfg)
            self._pipeline.start()
            logger.info("Pipeline started: project_id=%s", cfg.project_id)
        else:
            logger.info("No active project at startup.")
        self._event_dispatcher.subscribe(self.on_config_change)

    def stop(self) -> None:
        """
        Stop and dispose the running pipeline.
        """
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        """
        Dispatch incoming config change events to lifecycle / update handlers.
        """
        match event:
            case ProjectActivationEvent() as e:
                project_id = UUID(e.project_id)
                cfg = self._get_pipeline_config(project_id)
                if self._pipeline:
                    self._pipeline.stop()
                self._pipeline = Pipeline(pipeline_conf=cfg)
                self._pipeline.start()
                logger.info("Pipeline started for activated project %s", e.project_id)

            case ProjectDeactivationEvent() as e:
                if self._pipeline and str(self._pipeline.config.project_id) == e.project_id:
                    self._pipeline.stop()
                    self._pipeline = None
                    logger.info("Pipeline stopped due to project deactivation %s", e.project_id)

            case ComponentConfigChangeEvent() as e:
                if self._pipeline and str(self._pipeline.config.project_id) == e.project_id:
                    new_cfg = self._get_pipeline_config(self._pipeline.config.project_id)
                    self._pipeline.update_config(new_cfg)
                    logger.info("Pipeline config updated for project %s", e.project_id)


    def _get_active_pipeline_config(self) -> PipelineConfig | None:
        """
        Build config for the currently active project in a single service call.
        Returns None if no active project.
        """
        with self._project_service() as svc:
            try:
                runtime_cfg = svc.get_active_project_runtime_config()
            except ResourceNotFoundError:
                return None
        return self._assemble_pipeline_config(runtime_cfg)

    def _get_pipeline_config(self, project_id: UUID) -> PipelineConfig:
        """
        Build config for a specific project.
        """
        with self._project_service() as svc:
            runtime_cfg = svc.get_project_runtime_config(project_id)
        return self._assemble_pipeline_config(runtime_cfg)

    def _assemble_pipeline_config(self, runtime_cfg: ProjectRuntimeConfig) -> PipelineConfig:
        """
        Convert full runtime project config into a PipelineConfig.
        """
        reader = self._select_reader(runtime_cfg)
        processor = self._select_processor(runtime_cfg)
        writer = self._select_writer(runtime_cfg)
        return PipelineConfig(
            project_id=runtime_cfg.id,
            reader=reader,
            processor=processor,
            writer=writer,
        )

    def _select_reader(self, runtime_cfg: ProjectRuntimeConfig) -> ReaderConfig:
        connected = next((src.config for src in runtime_cfg.sources if src.connected), None)
        if connected:
            return connected
        if runtime_cfg.sources:
            return runtime_cfg.sources[0].config
        return WebCamConfig(source_type=SourceType.WEBCAM, device_id=0)

    def _select_processor(self, runtime_cfg: ProjectRuntimeConfig) -> ModelConfig:
        return runtime_cfg.processors[0] if runtime_cfg.processors else ModelConfig()

    def _select_writer(self, runtime_cfg: ProjectRuntimeConfig) -> WriterConfig:
        return runtime_cfg.sinks[0] if runtime_cfg.sinks else WriterConfig()
