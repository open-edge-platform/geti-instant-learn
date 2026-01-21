#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import queue
import threading
from uuid import UUID

import cv2
from getiprompt.data.base.batch import Batch
from sqlalchemy.orm import Session, sessionmaker

from domain.db.models import PromptType
from domain.dispatcher import (
    ComponentConfigChangeEvent,
    ComponentType,
    ConfigChangeDispatcher,
    ConfigChangeEvent,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from domain.repositories.frame import FrameRepository
from domain.repositories.prompt import PromptRepository
from domain.services.label import LabelService
from domain.services.project import ProjectService
from domain.services.schemas.label import VisualizationInfo
from domain.services.schemas.mappers.prompt import visual_prompt_to_sample
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.processor import InputData, OutputData
from domain.services.schemas.reader import FrameListResponse
from runtime.components import ComponentFactory, DefaultComponentFactory
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.errors import UnsupportedOperationError
from runtime.core.components.pipeline import Pipeline
from runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError, SourceNotSeekableError

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
        self._frame_repository = FrameRepository()
        self._component_factory = component_factory or DefaultComponentFactory(session_factory)
        # todo: bundle refs to pipeline and pipeline config together.
        self._pipeline: Pipeline | None = None
        self._current_config: PipelineConfig | None = None
        self._visualization_info: VisualizationInfo | None = None
        self._visualization_lock = threading.Lock()

    def start(self) -> None:
        """
        Start pipeline for active project if present; subscribe to config events.
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
            cfg = svc.get_active_pipeline_config()
        if cfg:
            self._current_config = cfg
            self._pipeline = self._create_pipeline(cfg.project_id)
            self._refresh_visualization_info(cfg.project_id)
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

    def get_visualization_info(self, project_id: UUID) -> VisualizationInfo | None:
        """
        Get cached visualization info for the active pipeline.
        """
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError(
                f"Project ID {project_id} does not match the active pipeline's project ID."
            )
        with self._visualization_lock:
            return self._visualization_info

    def _refresh_visualization_info(self, project_id: UUID) -> None:
        """
        Refresh cached visualization info from a database.

        Called when a pipeline starts or prompts/labels change.
        """
        with self._session_factory() as session:
            label_svc = LabelService(session=session)
            prompt_repo = PromptRepository(session=session)

            vis_labels = label_svc.get_visualization_labels(project_id)
            prompts = prompt_repo.list_all_by_project(project_id=project_id, prompt_type=PromptType.VISUAL)
            all_label_ids: set[UUID] = set()
            for prompt in prompts:
                all_label_ids.update(ann.label_id for ann in prompt.annotations)

            category_mappings = label_svc.build_category_mappings(all_label_ids)

        with self._visualization_lock:
            self._visualization_info = VisualizationInfo(
                label_colors=vis_labels,
                category_mappings=category_mappings,
            )
        logger.debug("Refreshed visualization info for project %s", project_id)

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        """
        React to configuration change events.
        """
        match event:
            case ProjectActivationEvent() as e:
                if self._pipeline:
                    self._pipeline.stop()
                self._pipeline = self._create_pipeline(e.project_id)
                self._refresh_visualization_info(e.project_id)
                self._pipeline.start()
                logger.info("Pipeline started for activated project %s", e.project_id)

            case ProjectDeactivationEvent() as e:
                if self._pipeline and self._pipeline.project_id == e.project_id:
                    self._pipeline.stop()
                    self._current_config = None
                    with self._visualization_lock:
                        self._visualization_info = None
                    logger.info("Pipeline stopped due to project deactivation %s", e.project_id)

            case ComponentConfigChangeEvent() as e:
                if self._pipeline and self._pipeline.project_id == e.project_id:
                    self._update_pipeline_components(e.project_id, e.component_type)
                    if e.component_type == ComponentType.PROCESSOR:
                        self._refresh_visualization_info(e.project_id)
                    logger.info("Pipeline components updated for project %s", e.project_id)

    def _create_pipeline(self, project_id: UUID) -> Pipeline:
        """
        Create a new Pipeline instance with components built from the given configuration.

        Args:
            config: The pipeline configuration.

        Returns:
            A fully initialized Pipeline instance (not yet started).
        """
        source = self._component_factory.create_source(project_id)
        reference_batch, category_id_to_label_id = self.get_reference_batch(project_id, PromptType.VISUAL) or (None, {})
        processor = self._component_factory.create_processor(project_id, reference_batch, category_id_to_label_id)
        sink = self._component_factory.create_sink(project_id)

        return (
            Pipeline(
                project_id,
                self._frame_repository,
                FrameBroadcaster[InputData](),
                FrameBroadcaster[OutputData](),
            )
            .set_source(source)
            .set_processor(processor)
            .set_sink(sink)
        )

    def _update_pipeline_components(self, project_id: UUID, component_type: ComponentType) -> None:
        """
        Compare current and new configurations, updating only changed components.

        Args:
            project_id: The project ID for the pipeline.
            component_type: The type of component to update.
        """
        if not self._pipeline:
            return

        match component_type:
            case ComponentType.SOURCE:
                source = self._component_factory.create_source(project_id)
                self._pipeline.set_source(source, True)
            case ComponentType.PROCESSOR:
                reference_batch, category_id_to_label_id = self.get_reference_batch(project_id, PromptType.VISUAL) or (
                    None,
                    {},
                )
                processor = self._component_factory.create_processor(
                    project_id, reference_batch, category_id_to_label_id
                )
                self._pipeline.set_processor(processor, True)
            case ComponentType.SINK:
                sink = self._component_factory.create_sink(project_id)
                self._pipeline.set_sink(sink, True)
            case _ as unknown:
                logger.error(f"Unknown component type {unknown}")

    # todo:
    # 1. unify methods for registering/unregistering all types of consumers.
    # 2. use context manager to automatically unregister a queue when it exits the scope

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

    def seek(self, project_id: UUID, index: int) -> None:
        """
        Seek to a specific frame in the active pipeline's source.

        Args:
            project_id: The project ID to verify against the active pipeline.
            index: The target frame index.

        Raises:
            PipelineNotActiveError: If no pipeline is running.
            PipelineProjectMismatchError: If project_id doesn't match the active pipeline.
            SourceNotSeekableError: If the source doesn't support seeking.
            IndexError: If index is out of bounds.
        """
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError(
                f"Project ID {project_id} does not match the active pipeline's project ID."
            )
        try:
            self._pipeline.seek(index)
        except UnsupportedOperationError:
            raise SourceNotSeekableError("The active source does not support frame navigation.")

    def get_frame_index(self, project_id: UUID) -> int:
        """
        Get the current frame index from the active pipeline's source.

        Args:
            project_id: The project ID to verify against the active pipeline.

        Returns:
            The current frame index.

        Raises:
            PipelineNotActiveError: If no pipeline is running.
            PipelineProjectMismatchError: If project_id doesn't match the active pipeline.
            SourceNotSeekableError: If the source doesn't support indexing.
        """
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError(
                f"Project ID {project_id} does not match the active pipeline's project ID."
            )
        try:
            return self._pipeline.get_frame_index()
        except UnsupportedOperationError:
            raise SourceNotSeekableError("The active source does not support frame indexing.")

    def list_frames(self, project_id: UUID, offset: int = 0, limit: int = 30) -> FrameListResponse:
        """
        Get a paginated list of frames from the active pipeline's source.

        Args:
            project_id: The project ID to verify against the active pipeline.
            offset: Number of items to skip (0-based index).
            limit: Maximum number of frames to return.

        Returns:
            FrameListResponse with frame metadata.

        Raises:
            PipelineNotActiveError: If no pipeline is running.
            PipelineProjectMismatchError: If project_id doesn't match the active pipeline.
            SourceNotSeekableError: If the source doesn't support frame listing.
        """
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError(
                f"Project ID {project_id} does not match the active pipeline's project ID."
            )
        try:
            return self._pipeline.list_frames(offset, limit)
        except UnsupportedOperationError:
            raise SourceNotSeekableError("The active source does not support frame listing.")

    def capture_frame(self, project_id: UUID) -> UUID:
        """
        Capture the latest frame from the active pipeline.

        Args:
            project_id: The project ID.

        Returns:
            UUID of the captured frame.
        """
        if self._pipeline is None:
            raise PipelineNotActiveError("No active pipeline.")
        if project_id != self._pipeline.project_id:
            raise PipelineProjectMismatchError(
                f"Project ID {project_id} does not match the active pipeline's project ID."
            )
        return self._pipeline.capture_frame()

    def get_reference_batch(self, project_id: UUID, prompt_type: PromptType) -> tuple[Batch, dict[int, str]] | None:
        """
        Get all prompts of a specific type for a project, formatted for model training.

        Returns:
            Tuple of (Batch, category_id_to_label_id mapping), or None if no valid samples were found.
        """
        if prompt_type == PromptType.TEXT:
            logger.warning("Text prompts not supported for training data generation: project_id=%s", project_id)
            return None

        with self._session_factory() as session:
            prompt_repo = PromptRepository(session=session)
            label_svc = LabelService(session=session)

            db_prompts = prompt_repo.list_all_by_project(project_id=project_id, prompt_type=prompt_type)
            if not db_prompts:
                logger.info("No prompts found for project_id=%s, prompt_type=%s", project_id, prompt_type)
                return None

            all_label_ids: set[UUID] = set()
            for prompt in db_prompts:
                all_label_ids.update(ann.label_id for ann in prompt.annotations)

            category_mappings = label_svc.build_category_mappings(all_label_ids)

            # track shot counts across prompts
            label_shot_counts: dict[UUID, int] = {}
            samples = []

            for prompt in db_prompts:
                if not prompt.frame_id:
                    logger.warning("Visual prompt missing frame_id: prompt_id=%s", prompt.id)
                    continue

                try:
                    frame = self._frame_repository.read_frame(project_id, prompt.frame_id)
                    if frame is None:
                        logger.warning("Frame not found: prompt_id=%s, frame_id=%s", prompt.id, prompt.frame_id)
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sample = visual_prompt_to_sample(
                        prompt, frame_rgb, category_mappings.label_to_category_id, label_shot_counts
                    )
                    samples.append(sample)

                except Exception as e:
                    logger.warning("Failed to convert prompt: prompt_id=%s, error=%s", prompt.id, e)
                    continue

            if not samples:
                logger.info("No valid samples generated: project_id=%s", project_id)
                return None

            batch = Batch.collate(samples)
            logger.debug("Reference batch: %s", batch)
            shots_per_category = {
                category_id: label_shot_counts.get(label_id, 0)
                for label_id, category_id in category_mappings.label_to_category_id.items()
            }
            logger.info(
                "Created reference batch: project_id=%s, samples=%d, categories=%d, shots_per_category=%s",
                project_id,
                len(batch.samples),
                len(category_mappings.label_to_category_id),
                shots_per_category,
            )
            return batch, category_mappings.category_id_to_label_id
