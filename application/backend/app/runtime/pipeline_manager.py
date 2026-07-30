#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import threading
from collections.abc import Callable
from uuid import UUID

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
from domain.services.schemas.label import CategoryMappings, VisualizationInfo
from domain.services.schemas.model_status import ModelStatus, ModelStatusErrorType, ModelStatusSchema
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.processor import (
    ErrorData,
    InputData,
    OutputData,
)
from domain.services.schemas.reader import FrameListResponse
from runtime.components import ComponentFactory, DefaultComponentFactory
from runtime.core.components.broadcaster import FrameBroadcaster, FrameSlot
from runtime.core.components.errors import UnsupportedOperationError
from runtime.core.components.pipeline import Pipeline
from runtime.core.components.processor import Processor
from runtime.errors import (
    PipelineNotActiveError,
    PipelineProjectMismatchError,
    PipelineReloadInProgressError,
    SourceNotSeekableError,
)
from runtime.services.model_load_error import model_load_error
from runtime.services.reference_batch import ReferenceBatchService

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
        self._component_factory = component_factory or DefaultComponentFactory()
        self._batch_service = ReferenceBatchService(session_factory, self._frame_repository)
        # todo: bundle refs to pipeline and pipeline config together.
        self._pipeline: Pipeline | None = None
        self._current_config: PipelineConfig | None = None
        self._visualization_info: VisualizationInfo | None = None
        self._lock = threading.RLock()
        # Serializes the long-running model builds. Kept separate from ``_lock``
        # so building a model never blocks the fast state accessors.
        self._build_lock = threading.Lock()
        # Bumped on every teardown so a build that finished after the pipeline
        # was torn down can be discarded instead of resurrecting it.
        self._build_generation = 0  #todo wtf?
        self._model_status: ModelStatusSchema | None = None

    def is_model_loading(self) -> bool:
        """Return True while a processor (re)build is in progress."""
        return self._model_status is not None and self._model_status.status == ModelStatus.LOADING

    def get_model_status(self) -> ModelStatusSchema:
        """Return the current processor load status and the last load error, if any."""
        if self._model_status is None:
            return ModelStatusSchema()
        return self._model_status.model_copy(deep=True)

    def _set_model_status(
        self,
        status: ModelStatus,
        error_type: ModelStatusErrorType | None = None,
        error_message: str | None = None,
        error_doc_url: str | None = None,
    ) -> None:
        self._model_status = ModelStatusSchema(
            status=status,
            error_type=error_type,
            error_message=error_message,
            error_doc_url=error_doc_url,
        )

    def reload_pipeline(self, project_id: UUID) -> None:
        """Stop and fully rebuild the active pipeline for the given project."""
        if self.is_model_loading():
            raise PipelineReloadInProgressError("Pipeline reload is already in progress.")
        self._teardown_pipeline()
        try:
            self._build_and_start_pipeline(project_id)
        except Exception:
            logger.exception("Pipeline reload failed for project %s", project_id)
        else:
            logger.info("Pipeline reloaded for project %s", project_id)

    def start(self) -> None:
        """
        Start pipeline for active project if present; subscribe to config events.
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
            cfg = svc.get_active_pipeline_config()
        if cfg:
            try:
                self._build_and_start_pipeline(cfg.project_id)
            except Exception:
                logger.exception("Pipeline startup failed for project %s", cfg.project_id)
            else:
                logger.info("Pipeline started: project_id=%s", cfg.project_id)
        else:
            logger.info("No active project found at startup.")
        self._event_dispatcher.subscribe(self.on_config_change)

    def stop(self) -> None:
        """Stop and dispose the running pipeline."""
        self._teardown_pipeline()

    def _teardown_pipeline(self) -> None:
        """Stop and clear the active pipeline and its associated state."""
        with self._lock:
            if self._pipeline:
                self._pipeline.stop()
                self._pipeline = None
            self._current_config = None
            self._visualization_info = None
            self._build_generation += 1

    def get_visualization_info(self, project_id: UUID) -> VisualizationInfo | None:
        """Get cached visualization info for the active pipeline."""
        with self._lock:
            if self._pipeline is None:
                raise PipelineNotActiveError("No active pipeline.")
            if project_id != self._pipeline.project_id:
                raise PipelineProjectMismatchError(
                    f"Project ID {project_id} does not match the active pipeline's project ID."
                )
            return self._visualization_info

    def _refresh_visualization_info(self, project_id: UUID) -> None:
        """
        Refresh cached visualization info from a database.
        Called when a pipeline starts or prompts/labels change. Must be called while self._lock is held.
        """
        with self._session_factory() as session:
            label_svc = LabelService(session=session)
            prompt_repo = PromptRepository(session=session)

            vis_labels = label_svc.get_visualization_labels(project_id)

            prompt_mode = self._current_config.prompt_mode if self._current_config else PromptType.VISUAL

            if prompt_mode == PromptType.TEXT:
                text_prompts = prompt_repo.list_by_project_and_type(project_id=project_id, prompt_type=PromptType.TEXT)
                text_categories = {idx: prompt.text for idx, prompt in enumerate(text_prompts) if prompt.text}
                empty_mappings = CategoryMappings(label_to_category_id={}, category_id_to_label_id={})
                self._visualization_info = VisualizationInfo(
                    label_colors=vis_labels,
                    category_mappings=empty_mappings,
                    text_categories=text_categories,
                )
            else:
                prompts = prompt_repo.list_by_project_and_type(project_id=project_id, prompt_type=PromptType.VISUAL)
                all_label_ids: set[UUID] = set()
                for prompt in prompts:
                    all_label_ids.update(ann.label_id for ann in prompt.annotations)

                category_mappings = label_svc.build_category_mappings(all_label_ids)
                self._visualization_info = VisualizationInfo(
                    label_colors=vis_labels, category_mappings=category_mappings
                )

        logger.debug("Refreshed visualization info for project %s", project_id)

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        """React to configuration change events.

        Model builds are intentionally performed without holding ``_lock`` so a
        slow export never stalls WebRTC connections or status polling.
        """
        match event:
            case ProjectActivationEvent() as e:
                self._teardown_pipeline()
                self._build_and_start_pipeline(e.project_id)
                logger.info("Pipeline started for activated project %s", e.project_id)

            case ProjectDeactivationEvent() as e:
                if self._is_active_project(e.project_id):
                    self._teardown_pipeline()
                    logger.info("Pipeline stopped due to project deactivation %s", e.project_id)

            case ComponentConfigChangeEvent() as e:
                if self._is_active_project(e.project_id):
                    self._update_pipeline_components(e.project_id, e.component_type)
                    if e.component_type == ComponentType.PROCESSOR:
                        self._refresh_visualization_info(e.project_id)
                    logger.info("Pipeline components updated for project %s", e.project_id)

    def _is_active_project(self, project_id: UUID) -> bool:
        """Return True when a pipeline is running for *project_id*."""
        with self._lock:
            return self._pipeline is not None and self._pipeline.project_id == project_id

    def _build_and_start_pipeline(self, project_id: UUID) -> None:
        """
        Create and start a new pipeline for the given project.

        The pipeline is built outside ``_lock`` (model loading can take minutes)
        and only installed if no teardown happened in the meantime.
        """
        with self._build_lock:
            with self._lock:
                generation = self._build_generation

            pipeline = self._create_pipeline(project_id)

            with self._lock:
                if generation != self._build_generation:
                    logger.info("Discarding stale pipeline build for project %s", project_id)
                    pipeline.stop()
                    return
                self._pipeline = pipeline
                self._refresh_visualization_info(project_id)
                self._pipeline.start()

    def _create_pipeline(self, project_id: UUID) -> Pipeline:
        """
        Create a new Pipeline instance with components from the given configuration.

        The processor (model download, OpenVINO export, and fit) is built *outside*
        ``self._lock``: it can take minutes, and holding the lock would block
        every other manager call, including ``get_output_slot()`` which is invoked
        from the asyncio event loop when a WebRTC client connects.

        If processor creation fails, records ERROR status and falls back to a
        PassThroughModelHandler so the pipeline can still start.

        Returns:
            A fully initialized Pipeline instance (not yet started).
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session)
            cfg = svc.get_pipeline_config(project_id)
        self._current_config = cfg
        source = self._component_factory.create_source(cfg.reader)
        processor = self._build_processor(cfg, project_id, fallback_to_passthrough=True)
        sink = self._component_factory.create_sink(cfg.writer)

        return (
            Pipeline(
                project_id,
                self._frame_repository,
                FrameBroadcaster[InputData | ErrorData]("inbound"),
                FrameBroadcaster[OutputData | ErrorData]("outbound"),
            )
            .set_source(source)
            .set_processor(processor)
            .set_sink(sink)
        )

    def _build_processor(self, cfg: PipelineConfig, project_id: UUID, fallback_to_passthrough: bool) -> Processor:
        """Build the processor and track its load status.

        The model is fully constructed, fitted and (when OpenVINO is enabled)
        exported here, so ``ModelStatus.READY`` is only reported once inference
        can actually run.

        Args:
            cfg: Pipeline configuration to build from.
            project_id: Project the processor belongs to, for logging.
            fallback_to_passthrough: When True, a build failure yields a
                pass-through processor so the pipeline still streams raw frames.
                When False, the error is re-raised to the caller.

        Returns:
            A processor ready to be installed in the pipeline.
        """
        self._set_model_status(ModelStatus.LOADING)
        try:
            reference_batch, _ = self._batch_service.build(cfg) or (None, {})
            processor = self._component_factory.create_processor(cfg, reference_batch)
        except Exception as exc:
            error_type, error_message, error_doc_url = model_load_error(exc)
            self._set_model_status(
                ModelStatus.ERROR, error_type=error_type, error_message=error_message, error_doc_url=error_doc_url
            )
            if not fallback_to_passthrough:
                logger.exception("Processor rebuild failed for project %s", project_id)
                raise
            logger.exception("Processor failed for project %s, falling back to passthrough", project_id)
            return self._component_factory.create_processor(cfg, None)
        self._set_model_status(ModelStatus.READY)
        return processor

    def _update_pipeline_components(self, project_id: UUID, component_type: ComponentType) -> None:
        """
        Compare current and new configurations, updating only changed components.

        Component construction happens outside ``_lock``; the lock is only taken
        to swap the finished component into the running pipeline.

        Args:
            project_id: The project ID for the pipeline.
            component_type: The type of component to update.
        """
        with self._build_lock:
            if not self._is_active_project(project_id):
                return

            with self._session_factory() as session:
                svc = ProjectService(session=session)
                cfg = svc.get_pipeline_config(project_id)
            with self._lock:
                self._current_config = cfg

            match component_type:
                case ComponentType.SOURCE:
                    source = self._component_factory.create_source(cfg.reader)
                    self._swap_component(lambda pipeline: pipeline.set_source(source, True))
                case ComponentType.PROCESSOR:
                    # Building the reference batch + downloading weights + initializing the model
                    # can take a while. Surface a "busy" flag so the UI can show a blocking overlay.
                    processor = self._build_processor(cfg, project_id, fallback_to_passthrough=False)
                    self._swap_component(lambda pipeline: pipeline.set_processor(processor, True))
                case ComponentType.SINK:
                    sink = self._component_factory.create_sink(cfg.writer)
                    self._swap_component(lambda pipeline: pipeline.set_sink(sink, True))
                case _ as unknown:
                    logger.error(f"Unknown component type {unknown}")

    def _swap_component(self, install: Callable[[Pipeline], None]) -> None:
        """Install a freshly built component into the active pipeline, if any."""
        with self._lock:
            if self._pipeline is None:
                logger.info("Pipeline was deleted while building a component, discarding it")
                return
            install(self._pipeline)

    def get_output_slot(self, project_id: UUID) -> FrameSlot[OutputData]:
        """Get the shared output slot for reading the latest processed frame.

        External consumers (e.g. WebRTC streams) can poll this slot without
        registering or unregistering — they simply read ``slot.latest``.
        """
        with self._lock:
            if self._pipeline is None:
                raise PipelineNotActiveError("No active pipeline.")
            if project_id != self._pipeline.project_id:
                raise PipelineProjectMismatchError("Project ID does not match the active pipeline's project ID.")
            return self._pipeline.outbound_slot

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
