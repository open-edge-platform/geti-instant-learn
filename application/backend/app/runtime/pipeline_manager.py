#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import queue
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from domain.db.models import PromptType
from domain.dispatcher import (
    ComponentConfigChangeEvent,
    ConfigChangeDispatcher,
    ConfigChangeEvent,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from domain.services.project import ProjectService
from domain.services.prompt import PromptService
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.factories.components import ComponentFactory, DefaultComponentFactory
from runtime.core.components.pipeline import Pipeline
from runtime.core.components.schemas.pipeline import PipelineConfig
from runtime.core.components.schemas.processor import InputData, MatcherConfig, OutputData
from runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages the active Pipeline and its lifecycle, handling configuration changes.

    This class is responsible for:
    - Creating and managing the active Pipeline instance
    - Tracking the current pipeline configuration
    - Reacting to configuration change events and determining which components need updates
    - Creating new component instances and instructing the pipeline to update them

    The Pipeline itself only manages component lifecycle (start/stop/replace), while
    the PipelineManager handles the business logic of configuration comparison and
    component instantiation.
    """

    def __init__(
        self,
        event_dispatcher: ConfigChangeDispatcher,
        session_factory: sessionmaker[Session],
        component_factory: ComponentFactory | None = None,
    ):
        self._event_dispatcher = event_dispatcher
        self._session_factory = session_factory
        self._component_factory = component_factory or DefaultComponentFactory()
        # todo: bundle refs to pipeline and pipeline config together.
        self._pipeline: Pipeline | None = None
        self._current_config: PipelineConfig | None = None

    def start(self) -> None:
        """
        Start pipeline for active project if present; subscribe to config events.
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
            cfg = svc.get_active_pipeline_config()
        if cfg:
            self._current_config = cfg
            self._pipeline = self._create_pipeline(cfg)
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
        self._current_config = None

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        """
        React to configuration change events.
        """
        match event:
            case ProjectActivationEvent() as e:
                with self._session_factory() as session:
                    svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
                    cfg = svc.get_pipeline_config(e.project_id)
                if self._pipeline:
                    self._pipeline.stop()
                self._current_config = cfg
                self._pipeline = self._create_pipeline(cfg)
                self._pipeline.start()
                logger.info("Pipeline started for activated project %s", e.project_id)

            case ProjectDeactivationEvent() as e:
                if self._pipeline and self._pipeline.project_id == e.project_id:
                    self._pipeline.stop()
                    self._pipeline = None
                    self._current_config = None
                    logger.info("Pipeline stopped due to project deactivation %s", e.project_id)

            case ComponentConfigChangeEvent() as e:
                if self._pipeline and self._pipeline.project_id == e.project_id:
                    with self._session_factory() as session:
                        svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
                        new_cfg = svc.get_pipeline_config(self._pipeline.project_id)
                    self._update_pipeline_components(new_cfg)
                    self._current_config = new_cfg
                    logger.info("Pipeline components updated for project %s", e.project_id)

    def _create_pipeline(self, config: PipelineConfig) -> Pipeline:
        """
        Create a new Pipeline instance with components built from the given configuration.

        Args:
            config: The pipeline configuration.

        Returns:
            A fully initialized Pipeline instance (not yet started).
        """
        inbound_bcast = FrameBroadcaster[InputData]()
        outbound_bcast = FrameBroadcaster[OutputData]()

        with self._session_factory() as session:
            prompt_svc = PromptService(session)
            reference_batch = prompt_svc.get_training_data(config.project_id, PromptType.VISUAL)

        source = self._component_factory.create_source(config.reader, inbound_bcast)
        processor = self._component_factory.create_processor(
            inbound_bcast,
            outbound_bcast,
            config.processor,
            reference_batch,
        )
        sink = self._component_factory.create_sink(outbound_bcast, config.writer)

        return Pipeline(
            project_id=config.project_id,
            source=source,
            processor=processor,
            sink=sink,
            inbound_broadcaster=inbound_bcast,
            outbound_broadcaster=outbound_bcast,
        )

    def _update_pipeline_components(self, new_config: PipelineConfig) -> None:
        """
        Compare current and new configurations, updating only changed components.

        Args:
            new_config: The new pipeline configuration.
        """
        if not self._pipeline or not self._current_config:
            return

        if new_config.reader != self._current_config.reader:
            logger.info(f"Source config changed: {self._current_config.reader} -> {new_config.reader}")
            new_source = self._component_factory.create_source(new_config.reader, self._pipeline._inbound_broadcaster)
            self._pipeline.update_component(new_source)

        if new_config.processor != self._current_config.processor:
            logger.info(f"Processor config changed: {self._current_config.processor} -> {new_config.processor}")

            with self._session_factory() as session:
                prompt_svc = PromptService(session)
                reference_batch = prompt_svc.get_training_data(new_config.project_id, PromptType.VISUAL)

            new_processor = self._component_factory.create_processor(
                self._pipeline._inbound_broadcaster,
                self._pipeline._outbound_broadcaster,
                new_config.processor,
                reference_batch,
            )
            self._pipeline.update_component(new_processor)

        if new_config.writer != self._current_config.writer:
            logger.info(f"Sink config changed: {self._current_config.writer} -> {new_config.writer}")
            new_sink = self._component_factory.create_sink(self._pipeline._outbound_broadcaster, new_config.writer)
            self._pipeline.update_component(new_sink)

    # todo:
    # 1. unify methods for registring/unregistring all types of consumers.
    # 2. use context manager to automaticaly unregister a queue when it exists the scope

    def register_webrtc(self, project_id: UUID) -> queue.Queue:
        """Register webRTC in pipeline."""
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline to register to.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError("Project ID does not match the active pipeline's project ID.")
        return self._pipeline.register_webrtc()

    def unregister_webrtc(self, target_queue: queue.Queue, project_id: UUID) -> None:
        """Unregister webRTC in pipeline."""
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline to unregister from.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError("Project ID does not match the active pipeline's project ID.")
        return self._pipeline.unregister_webrtc(queue=target_queue)

    def register_inbound_consumer(self, project_id: UUID) -> queue.Queue[InputData]:
        """
        Register a consumer for raw input frames from the source.

        Args:
            project_id: The project ID to verify against the active pipeline.

        Returns:
            A queue that will receive raw input frames.

        Raises:
            PipelineNotActiveError: If no pipeline is running.
            PipelineProjectMismatchError: If project_id doesn't match the active pipeline.
        """
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline to register inbound consumer.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError(
                f"Project ID {project_id} does not match the active pipeline's project ID {self._pipeline.project_id}."
            )
        return self._pipeline.register_inbound_consumer()

    def unregister_inbound_consumer(self, project_id: UUID, target_queue: queue.Queue[InputData]) -> None:
        """
        Unregister a consumer for raw input frames.

        Args:
            project_id: The project ID to verify against the active pipeline.
            target_queue: The queue to unregister.

        Raises:
            PipelineNotActiveError: If no pipeline is running.
            PipelineProjectMismatchError: If project_id doesn't match the active pipeline.
        """
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline to unregister inbound consumer from.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError(
                f"Project ID {project_id} does not match the active pipeline's project ID {self._pipeline.project_id}."
            )
        self._pipeline.unregister_inbound_consumer(target_queue)
