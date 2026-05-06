#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
from typing import TYPE_CHECKING
from uuid import UUID

import cv2
from instantlearn.data.base.batch import Batch
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Callable

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
from domain.services.schemas.model_status import ModelStatusSchema
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.processor import InputData, OutputData
from domain.services.schemas.reader import FrameListResponse
from runtime.components import ComponentFactory, DefaultComponentFactory
from runtime.core.components.broadcaster import FrameBroadcaster, FrameSlot
from runtime.core.components.errors import UnsupportedOperationError
from runtime.core.components.model_status_reporter import ModelStatusReporter
from runtime.core.components.pipeline import Pipeline
from runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError, SourceNotSeekableError

logger = logging.getLogger(__name__)


class PublishingReporter(ModelStatusReporter):
    """Adapter that translates ``ModelStatusReporter`` calls from a ``Processor``
    into status snapshots published through ``PipelineManager._publish_status``.

    Each instance is bound to a single (project_id, model_name, device) descriptor
    so messages remain coherent even after the pipeline is replaced.
    """

    def __init__(
        self,
        publish: "Callable[[ModelStatusSchema], None]",
        project_id: UUID,
        model_name: str | None,
        device: str | None,
    ) -> None:
        self._publish = publish
        self._project_id = project_id
        self._model_name = model_name
        self._device = device

    def loading_model(self) -> None:
        self._publish(
            ModelStatusSchema.loading_model(
                project_id=self._project_id,
                model_name=self._model_name,
                device=self._device,
            )
        )

    def ready(self) -> None:
        self._publish(
            ModelStatusSchema.ready(
                project_id=self._project_id,
                model_name=self._model_name,
                device=self._device,
            )
        )

    def idle(self) -> None:
        self._publish(ModelStatusSchema.idle(project_id=self._project_id))

    def error(self, exc: BaseException) -> None:
        self._publish(
            ModelStatusSchema.from_exception(
                exc,
                project_id=self._project_id,
                model_name=self._model_name,
                device=self._device,
            )
        )


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

        # --- Model status broadcasting ---
        # Status transitions are emitted from worker threads (the dispatcher's
        # ThreadPoolExecutor) but consumed by SSE generators on the asyncio loop.
        # ``_loop`` is captured at FastAPI startup via ``bind_loop``.
        self._status: ModelStatusSchema = ModelStatusSchema.idle()
        self._status_lock = threading.Lock()
        self._subscribers: set[asyncio.Queue[ModelStatusSchema]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the asyncio event loop used to deliver status updates to SSE subscribers.

        Must be called from FastAPI's lifespan startup before the first config change.
        """
        self._loop = loop

    # ------------------------------------------------------------------
    # Model status: subscription & broadcasting
    # ------------------------------------------------------------------

    def subscribe_status(self) -> asyncio.Queue[ModelStatusSchema]:
        """Register a new SSE subscriber and return its queue."""
        queue: asyncio.Queue[ModelStatusSchema] = asyncio.Queue()
        with self._status_lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe_status(self, queue: asyncio.Queue[ModelStatusSchema]) -> None:
        """Remove a previously registered SSE subscriber."""
        with self._status_lock:
            self._subscribers.discard(queue)

    def get_status(self) -> ModelStatusSchema:
        """Return the current model status snapshot."""
        with self._status_lock:
            return self._status

    def _publish_status(self, status: ModelStatusSchema) -> None:
        """Update the cached status and fan out to all SSE subscribers."""
        with self._status_lock:
            self._status = status
            subscribers = list(self._subscribers)
        logger.info(
            "Model status: state=%s project_id=%s message=%s",
            status.state,
            status.project_id,
            status.message,
        )
        if not subscribers:
            return
        loop = self._loop
        if loop is None:
            # No async loop bound yet (e.g. during early startup). Snapshot is still cached.
            return
        for queue in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, status)

    def _resolve_model_descriptor(self, cfg: PipelineConfig | None) -> tuple[str | None, str | None]:
        """Extract a human-friendly model name and resolved device label from a pipeline config.

        The device is resolved through the component factory so ``auto`` and ``None``
        values get mapped to the concrete backend the model will actually use
        (e.g. ``xpu``/``cuda``/``cpu``), keeping status messages truthful.
        """
        if cfg is None:
            return None, None
        model_name: str | None = None
        if cfg.processor is not None:
            model_name = getattr(cfg.processor, "model_type", None)
            if model_name is not None:
                model_name = str(model_name)
        try:
            resolved = self._component_factory.resolve_device(cfg.device)
            device: str | None = str(resolved)
        except Exception:
            # Resolution failure (e.g. device probing error) shouldn't break status
            # reporting — fall back to the configured value or ``None``.
            logger.debug("Failed to resolve device for status descriptor", exc_info=True)
            device = str(cfg.device) if cfg.device is not None else None
        return model_name, device

    def _make_reporter(self, cfg: PipelineConfig) -> PublishingReporter:
        """Create a ``ModelStatusReporter`` bound to the given pipeline config."""
        model_name, device = self._resolve_model_descriptor(cfg)
        return PublishingReporter(
            publish=self._publish_status,
            project_id=cfg.project_id,
            model_name=model_name,
            device=device,
        )

    def _publish_error(self, cfg: PipelineConfig | None, exc: BaseException) -> None:
        """Publish an ERROR snapshot for failures that happen outside the processor thread."""
        model_name, device = self._resolve_model_descriptor(cfg)
        project_id = cfg.project_id if cfg is not None else None
        self._publish_status(
            ModelStatusSchema.from_exception(
                exc,
                project_id=project_id,
                model_name=model_name,
                device=device,
            )
        )

    def _publish_loading_reference_batch(self, cfg: PipelineConfig) -> None:
        """Publish a LOADING_REFERENCE_BATCH snapshot for manager-side prep work."""
        model_name, device = self._resolve_model_descriptor(cfg)
        self._publish_status(
            ModelStatusSchema.loading_reference_batch(
                project_id=cfg.project_id,
                model_name=model_name,
                device=device,
            )
        )

    def start(self) -> None:
        """
        Start pipeline for active project if present; subscribe to config events.
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
            cfg = svc.get_active_pipeline_config()
        if cfg:
            try:
                self._current_config = cfg
                self._pipeline = self._build_and_start_pipeline(cfg)
                logger.info("Pipeline started: project_id=%s", cfg.project_id)
            except Exception as exc:
                # Failure happened before the processor thread could take over the
                # status reporting (e.g. building the reference batch), so we
                # publish the ERROR snapshot from here.
                self._publish_error(cfg, exc)
                raise
        else:
            logger.info("No active project found at startup.")
            self._publish_status(ModelStatusSchema.idle())
        self._event_dispatcher.subscribe(self.on_config_change)

    def stop(self) -> None:
        """Stop and dispose the running pipeline.

        Always ends with an IDLE snapshot. The Processor thread may have already
        published IDLE from its ``_stop()``; re-publishing is idempotent from the
        UI's perspective and guarantees the UI recovers even if the pipeline stop
        raised or no processor was ever started.
        """
        pipeline = self._pipeline
        self._pipeline = None
        self._current_config = None
        try:
            if pipeline is not None:
                pipeline.stop()
        finally:
            self._publish_status(ModelStatusSchema.idle())

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
                with self._session_factory() as session:
                    svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
                    cfg = svc.get_pipeline_config(e.project_id)
                try:
                    self._current_config = cfg
                    self._pipeline = self._build_and_start_pipeline(cfg)
                    logger.info("Pipeline started for activated project %s", e.project_id)
                except Exception as exc:
                    self._publish_error(cfg, exc)
                    raise

            case ProjectDeactivationEvent() as e:
                if self._pipeline and self._pipeline.project_id == e.project_id:
                    # Processor._stop() publishes IDLE on its own when the model is closed.
                    self._pipeline.stop()
                    self._current_config = None
                    self._pipeline = None
                    with self._visualization_lock:
                        self._visualization_info = None
                    logger.info("Pipeline stopped due to project deactivation %s", e.project_id)

            case ComponentConfigChangeEvent() as e:
                if self._pipeline and self._pipeline.project_id == e.project_id:
                    self._update_pipeline_components(e.project_id, e.component_type)
                    if e.component_type == ComponentType.PROCESSOR:
                        self._refresh_visualization_info(e.project_id)
                    logger.info("Pipeline components updated for project %s", e.project_id)

    def _build_and_start_pipeline(self, cfg: PipelineConfig) -> Pipeline:
        """Create, configure and start a pipeline.

        Status emissions during this call are limited to ``LOADING_REFERENCE_BATCH``,
        which is work the manager performs synchronously before handing control to
        the processor thread. The processor itself emits ``LOADING_MODEL`` ->
        ``READY`` (or ``IDLE`` for passthrough) once it starts running, via the
        injected ``ModelStatusReporter``.
        """
        project_id = cfg.project_id
        reporter = self._make_reporter(cfg)

        # Manager-side work happens before the processor thread starts, so report it here.
        self._publish_loading_reference_batch(cfg)
        reference_batch, _category_id_to_label_id = self.get_reference_batch(project_id, PromptType.VISUAL) or (
            None,
            {},
        )

        source = self._component_factory.create_source(project_id)
        processor = self._component_factory.create_processor(project_id, reference_batch, status_reporter=reporter)
        sink = self._component_factory.create_sink(project_id)

        pipeline = (
            Pipeline(
                project_id,
                self._frame_repository,
                FrameBroadcaster[InputData]("inbound"),
                FrameBroadcaster[OutputData]("outbound"),
            )
            .set_source(source)
            .set_processor(processor)
            .set_sink(sink)
        )

        self._refresh_visualization_info(project_id)
        pipeline.start()
        # Processor emits LOADING_MODEL -> READY (or IDLE for passthrough) from its run loop.
        return pipeline

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
                # Re-read config to pick up any model/device changes for the status messages.
                with self._session_factory() as session:
                    svc = ProjectService(session=session)
                    cfg = svc.get_pipeline_config(project_id)
                self._current_config = cfg
                reporter = self._make_reporter(cfg)

                try:
                    self._publish_loading_reference_batch(cfg)
                    reference_batch, _category_id_to_label_id = self.get_reference_batch(
                        project_id, PromptType.VISUAL
                    ) or (None, {})

                    processor = self._component_factory.create_processor(
                        project_id, reference_batch, status_reporter=reporter
                    )
                except Exception as exc:
                    # Failure occurred before the new processor thread could take
                    # over status reporting, so publish ERROR from here.
                    self._publish_error(cfg, exc)
                    raise

                # Once set_processor is called the new processor thread owns status
                # reporting via the injected reporter — no manager-side ERROR publish
                # beyond this point.
                self._pipeline.set_processor(processor, True)
            case ComponentType.SINK:
                sink = self._component_factory.create_sink(project_id)
                self._pipeline.set_sink(sink, True)
            case _ as unknown:
                logger.error(f"Unknown component type {unknown}")

    def get_output_slot(self, project_id: UUID) -> FrameSlot[OutputData]:
        """Get the shared output slot for reading the latest processed frame.

        External consumers (e.g. WebRTC streams) can poll this slot without
        registering or unregistering — they simply read ``slot.latest``.
        """
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
