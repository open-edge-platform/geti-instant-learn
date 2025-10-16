#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from core.runtime.dispatcher import (
    ComponentConfigChangeEvent,
    ConfigChangeDispatcher,
    ConfigChangeEvent,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from core.runtime.pipeline import Pipeline
from services.project import ProjectService

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
        Context manager yielding a short-lived ProjectService, ensures session cleanup.
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
            yield svc

    def start(self) -> None:
        """
        Start pipeline for active project if present; subscribe to config events.
        """
        with self._project_service() as svc:
            cfg = svc.get_active_pipeline_config()
        if cfg:
            self._pipeline = Pipeline(pipeline_conf=cfg)
            self._pipeline.start()
            logger.info("Pipeline started: project_id=%s", cfg.project_id)
        else:
            logger.info("No active project found at startup.")
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
        React to configuration change events.
        """
        match event:
            case ProjectActivationEvent() as e:
                with self._project_service() as svc:
                    cfg = svc.get_pipeline_config(e.project_id)
                if self._pipeline:
                    self._pipeline.stop()
                self._pipeline = Pipeline(pipeline_conf=cfg)
                self._pipeline.start()
                logger.info("Pipeline started for activated project %s", e.project_id)

            case ProjectDeactivationEvent() as e:
                if self._pipeline and self._pipeline.config.project_id == e.project_id:
                    self._pipeline.stop()
                    self._pipeline = None
                    logger.info("Pipeline stopped due to project deactivation %s", e.project_id)

            case ComponentConfigChangeEvent() as e:
                if self._pipeline and self._pipeline.config.project_id == e.project_id:
                    with self._project_service() as svc:
                        new_cfg = svc.get_pipeline_config(self._pipeline.config.project_id)
                    self._pipeline.update_config(new_cfg)
                    logger.info("Pipeline config updated for project %s", e.project_id)
