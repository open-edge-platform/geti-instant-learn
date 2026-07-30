#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from domain.services.schemas.model_status import ModelStatus, ModelStatusSchema
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

# How long a reload waits for another lifecycle operation to finish before it
# reports a conflict (HTTP 409) instead of queueing behind it.
_RELOAD_LOCK_TIMEOUT_S = 2.0
# Upper bound for shutdown. ``create_processor`` can be an opaque, multi-minute call
# with no cancellation points, so ``stop()`` must not wait for it forever.
_SHUTDOWN_LOCK_TIMEOUT_S = 30.0


class _BuildCancelledError(Exception):
    """Raised internally when an in-flight build was cancelled by a teardown."""


@dataclass(frozen=True)
class _ActiveState:
    """Immutable snapshot of the running pipeline and everything derived from it.

    Bundling these together means a reader takes a single consistent snapshot
    instead of observing a half-updated combination of pipeline and config.
    """

    pipeline: Pipeline
    config: PipelineConfig
    visualization_info: VisualizationInfo | None = None


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

    Locking
    -------
    ``_lifecycle_lock`` serializes every mutating lifecycle operation (start,
    stop, reload, teardown, component updates). It is held for the whole
    duration of a model build, so lifecycle operations never interleave.

    ``_state_lock`` is short-lived and guards ``_state`` and ``_model_status``.
    It is never held across a model build or a DB query, so fast readers —
    ``get_output_slot()`` from the asyncio event loop, or ``/model-status``
    polling — are never blocked by a loading model.

    Because a teardown would otherwise have to wait for a running build, it
    first cancels the build via ``_active_build`` and only then takes the
    lifecycle lock. A cancelled build is discarded silently: it installs nothing
    and publishes no model status.
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
        self._state: _ActiveState | None = None
        # Re-entrant: a build may trigger a teardown on the same thread.
        self._lifecycle_lock = threading.RLock()
        self._state_lock = threading.RLock()
        # Cancellation token of the build currently in flight, if any.
        self._active_build: threading.Event | None = None
        self._model_status: ModelStatusSchema | None = None

    # ------------------------------------------------------------------
    # Model status (describes the processor of the active pipeline)
    # ------------------------------------------------------------------

    def is_model_loading(self) -> bool:
        """Return True while a processor (re)build is in progress."""
        with self._state_lock:
            return self._model_status is not None and self._model_status.status == ModelStatus.LOADING

    def get_model_status(self) -> ModelStatusSchema:
        """Return the current processor load status and the last load error, if any."""
        with self._state_lock:
            if self._model_status is None:
                return ModelStatusSchema()
            return self._model_status.model_copy(deep=True)

    def _mark_model_loading(self) -> None:
        """Announce that a processor build has begun."""
        self._store_model_status(ModelStatusSchema(status=ModelStatus.LOADING))

    def _store_model_status(self, status: ModelStatusSchema | None) -> None:
        with self._state_lock:
            self._model_status = status

    def _publish_processor_status(self, token: threading.Event, status: ModelStatusSchema) -> None:
        """Publish the terminal status of a processor build once it is installed.

        A cancelled build stays silent: whoever cancelled it owns the status
        (a teardown clears it, a rebuild sets it back to LOADING).
        """
        if token.is_set():
            return
        self._store_model_status(status)

    def _clear_loading_status(self) -> None:
        """Drop a dangling LOADING status when a build failed before installing."""
        with self._state_lock:
            if self._model_status is not None and self._model_status.status == ModelStatus.LOADING:
                self._model_status = None

    # ------------------------------------------------------------------
    # Build cancellation
    # ------------------------------------------------------------------

    def _new_build_token(self) -> threading.Event:
        with self._state_lock:
            token = threading.Event()
            self._active_build = token
            return token

    def _finish_build(self, token: threading.Event) -> None:
        """Retire a build token once its build reached a terminal state.

        Without this, a later teardown would "cancel" a build that finished long
        ago: harmless, but it makes the state impossible to reason about.
        """
        with self._state_lock:
            if self._active_build is token:
                self._active_build = None

    def _cancel_active_build(self) -> None:
        with self._state_lock:
            if self._active_build is not None:
                self._active_build.set()

    @staticmethod
    def _raise_if_cancelled(token: threading.Event) -> None:
        if token.is_set():
            raise _BuildCancelledError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload_pipeline(self, project_id: UUID) -> None:
        """Stop and fully rebuild the active pipeline for the given project.

        Blocks until the rebuild completed. Raises ``PipelineReloadInProgressError``
        when another lifecycle operation is already running.
        """
        if not self._lifecycle_lock.acquire(timeout=_RELOAD_LOCK_TIMEOUT_S):
            raise PipelineReloadInProgressError("Pipeline reload is already in progress.")
        try:
            self._restart_pipeline(project_id)
        except Exception:
            logger.exception("Pipeline reload failed for project %s", project_id)
        else:
            logger.info("Pipeline reloaded for project %s", project_id)
        finally:
            self._lifecycle_lock.release()

    def _restart_pipeline(self, project_id: UUID) -> None:
        """Replace the active pipeline with a freshly built one.

        Caller owns the lifecycle lock. LOADING is announced up front so the
        status never flickers back to "no processor" between disposing the old
        pipeline and building the new one.
        """
        self._mark_model_loading()
        self._teardown_pipeline(keep_status=True)
        self._build_and_start_pipeline(project_id)

    def start(self) -> None:
        """
        Start pipeline for active project if present; subscribe to config events.
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session, config_change_dispatcher=self._event_dispatcher)
            cfg = svc.get_active_pipeline_config()
        if cfg:
            with self._lifecycle_lock:
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
        """Stop and dispose the running pipeline.

        Shutdown must make progress even when a model build is stuck inside an
        uninterruptible call, so the lifecycle lock is only waited for briefly.
        """
        self._cancel_active_build()
        acquired = self._lifecycle_lock.acquire(timeout=_SHUTDOWN_LOCK_TIMEOUT_S)
        if not acquired:
            logger.warning("Lifecycle lock still busy at shutdown, disposing the pipeline without it.")
        try:
            self._dispose_active_state(keep_status=False)
        finally:
            if acquired:
                self._lifecycle_lock.release()

    def _teardown_pipeline(self, keep_status: bool = False) -> None:
        """Stop and clear the active pipeline and its associated state.

        Args:
            keep_status: Leave the model status untouched, for callers that have
                already announced the processor rebuild that follows.
        """
        # Cancel first, then wait for the lock: an in-flight build sees the token
        # at its next checkpoint and aborts instead of installing itself.
        self._cancel_active_build()
        with self._lifecycle_lock:
            self._dispose_active_state(keep_status=keep_status)

    def _dispose_active_state(self, keep_status: bool) -> None:
        """Drop the active state and stop its pipeline. Caller owns the lifecycle lock."""
        with self._state_lock:
            state = self._state
            self._state = None
            # Both callers cancelled the build first; its token is now spent.
            self._active_build = None
            if not keep_status:
                # The processor is gone, so its status must not outlive it and
                # leak into the next project.
                self._model_status = None
        if state is not None:
            state.pipeline.stop()

    def get_visualization_info(self, project_id: UUID) -> VisualizationInfo | None:
        """Get cached visualization info for the active pipeline."""
        return self._require_active_state(project_id).visualization_info

    def _refresh_visualization_info(self, project_id: UUID) -> None:
        """
        Refresh cached visualization info from a database.
        Called when a pipeline starts or prompts/labels change. The database work
        runs without ``_state_lock`` held; only the final swap takes it.
        """
        with self._state_lock:
            cfg = self._state.config if self._state is not None else None
        info = self._load_visualization_info(project_id, cfg)
        with self._state_lock:
            if self._state is not None and self._state.pipeline.project_id == project_id:
                self._state = replace(self._state, visualization_info=info)

    def _load_visualization_info(self, project_id: UUID, cfg: PipelineConfig | None) -> VisualizationInfo:
        """Read visualization info from the database. Must not be called under a lock."""
        with self._session_factory() as session:
            label_svc = LabelService(session=session)
            prompt_repo = PromptRepository(session=session)

            vis_labels = label_svc.get_visualization_labels(project_id)

            prompt_mode = cfg.prompt_mode if cfg else PromptType.VISUAL

            if prompt_mode == PromptType.TEXT:
                text_prompts = prompt_repo.list_by_project_and_type(project_id=project_id, prompt_type=PromptType.TEXT)
                text_categories = {idx: prompt.text for idx, prompt in enumerate(text_prompts) if prompt.text}
                empty_mappings = CategoryMappings(label_to_category_id={}, category_id_to_label_id={})
                info = VisualizationInfo(
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
                info = VisualizationInfo(label_colors=vis_labels, category_mappings=category_mappings)

        logger.debug("Refreshed visualization info for project %s", project_id)
        return info

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        """React to configuration change events.

        Model builds are performed without holding ``_state_lock`` so a slow
        export never stalls WebRTC connections or status polling.
        """
        match event:
            case ProjectActivationEvent() as e:
                with self._lifecycle_lock:
                    self._restart_pipeline(e.project_id)
                    logger.info("Pipeline started for activated project %s", e.project_id)

            case ProjectDeactivationEvent() as e:
                # Cheap early-out so an unrelated project never queues behind a build.
                if not self._is_active_project(e.project_id):
                    return
                with self._lifecycle_lock:
                    # Re-check: the pipeline may have changed while waiting.
                    if self._is_active_project(e.project_id):
                        self._teardown_pipeline()
                        logger.info("Pipeline stopped due to project deactivation %s", e.project_id)

            case ComponentConfigChangeEvent() as e:
                # Cheap early-out so an unrelated project never queues behind a build.
                if not self._is_active_project(e.project_id):
                    return
                with self._lifecycle_lock:
                    # _update_pipeline_components re-checks under this same lock.
                    self._update_pipeline_components(e.project_id, e.component_type)
                    if e.component_type == ComponentType.PROCESSOR:
                        self._refresh_visualization_info(e.project_id)
                    logger.info("Handled component config change for project %s", e.project_id)

    def _is_active_project(self, project_id: UUID) -> bool:
        """Return True when a pipeline is running for *project_id*."""
        with self._state_lock:
            return self._state is not None and self._state.pipeline.project_id == project_id

    def _build_and_start_pipeline(self, project_id: UUID) -> None:
        """
        Create and start a new pipeline for the given project.

        The pipeline is built while holding the lifecycle lock but *not* the
        state lock, so status polling and WebRTC connections stay responsive.
        A build cancelled by a concurrent teardown is discarded silently.
        """
        with self._lifecycle_lock:
            token = self._new_build_token()
            try:
                self._build_and_install(project_id, token)
            except _BuildCancelledError:
                logger.info("Discarding cancelled pipeline build for project %s", project_id)
            finally:
                self._finish_build(token)

    def _build_and_install(self, project_id: UUID, token: threading.Event) -> None:
        """Build a pipeline and install it as the active one. Caller owns the lifecycle lock.

        Raises:
            _BuildCancelledError: The build was cancelled and nothing was installed.
        """
        try:
            state, terminal_status = self._create_pipeline(project_id, token)
        except _BuildCancelledError:
            # Not a no-op: a cancellation must skip the status clean-up below,
            # because whoever cancelled this build owns the status.
            raise
        except Exception:
            self._clear_loading_status()
            raise

        try:
            self._raise_if_cancelled(token)
            state = replace(state, visualization_info=self._load_visualization_info(project_id, state.config))
            with self._state_lock:
                # Install and start atomically: a teardown landing between the
                # two would otherwise leave a running pipeline behind, and a
                # cancelled build must never transition one to running.
                self._raise_if_cancelled(token)
                self._state = state
                state.pipeline.start()
        except _BuildCancelledError:
            self._abandon_built_state(state)
            raise
        except Exception:
            self._clear_loading_status()
            self._abandon_built_state(state)
            raise

        self._publish_processor_status(token, terminal_status)

    def _abandon_built_state(self, state: _ActiveState) -> None:
        """Stop a freshly built pipeline, and unset it if it did become active."""
        with self._state_lock:
            if self._state is state:
                self._state = None
        state.pipeline.stop()

    def _create_pipeline(self, project_id: UUID, token: threading.Event) -> tuple[_ActiveState, ModelStatusSchema]:
        """
        Create a new Pipeline instance with components from the given configuration.

        The processor (model download, OpenVINO export, and fit) is built *outside*
        ``self._state_lock``: it can take minutes, and holding that lock would block
        every other manager call, including ``get_output_slot()`` which is invoked
        from the asyncio event loop when a WebRTC client connects.

        If processor creation fails, falls back to a PassThroughModelHandler so the
        pipeline can still start, and reports ERROR as the terminal status.

        Returns:
            The not-yet-installed state, and the status to publish once installed.
        """
        with self._session_factory() as session:
            svc = ProjectService(session=session)
            cfg = svc.get_pipeline_config(project_id)
        source = self._component_factory.create_source(cfg.reader)
        processor, terminal_status = self._build_processor(cfg, project_id, token, fallback_to_passthrough=True)
        sink = self._component_factory.create_sink(cfg.writer)

        pipeline = (
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
        return _ActiveState(pipeline=pipeline, config=cfg), terminal_status

    def _build_processor(
        self,
        cfg: PipelineConfig,
        project_id: UUID,
        token: threading.Event,
        fallback_to_passthrough: bool,
    ) -> tuple[Processor, ModelStatusSchema]:
        """Build the processor and compute its load status.

        The model is fully constructed, fitted and (when OpenVINO is enabled)
        exported here. A terminal status is only made visible once it is final:
        ``READY`` is published by the caller after the processor is installed,
        while ``ERROR`` is published here, on the paths where nothing will be
        installed at all.

        Args:
            cfg: Pipeline configuration to build from.
            project_id: Project the processor belongs to, for logging.
            token: Cancellation token of the current build.
            fallback_to_passthrough: When True, a build failure yields a
                pass-through processor so the pipeline still streams raw frames.
                When False, the error is re-raised to the caller.

        Returns:
            A processor ready to be installed, and its terminal status.

        Raises:
            _BuildCancelledError: The build was cancelled by a teardown.
        """
        self._mark_model_loading()
        self._raise_if_cancelled(token)
        try:
            reference_batch, _ = self._batch_service.build(cfg) or (None, {})
            self._raise_if_cancelled(token)
            processor = self._component_factory.create_processor(cfg, reference_batch)
        except _BuildCancelledError:
            # Not a no-op: a cancellation must not be classified as a load failure.
            raise
        except Exception as exc:
            # A failure caused by a concurrent teardown must stay silent.
            self._raise_if_cancelled(token)
            error_type, error_message, error_doc_url = model_load_error(exc)
            failed_status = ModelStatusSchema(
                status=ModelStatus.ERROR,
                error_type=error_type,
                error_message=error_message,
                error_doc_url=error_doc_url,
            )
            if not fallback_to_passthrough:
                logger.exception("Processor rebuild failed for project %s", project_id)
                # Nothing will be installed, so publish the failure right away.
                self._store_model_status(failed_status)
                raise
            logger.exception("Processor failed for project %s, falling back to passthrough", project_id)
            try:
                return self._component_factory.create_processor(cfg, None), failed_status
            except Exception:
                logger.exception("Passthrough processor creation failed for project %s", project_id)
                self._store_model_status(failed_status)
                raise
        self._raise_if_cancelled(token)
        return processor, ModelStatusSchema(status=ModelStatus.READY)

    def _update_pipeline_components(self, project_id: UUID, component_type: ComponentType) -> None:
        """
        Compare current and new configurations, updating only changed components.

        Component construction happens outside ``_state_lock``; that lock is only
        taken to swap the finished component into the running pipeline.

        Args:
            project_id: The project ID for the pipeline.
            component_type: The type of component to update.
        """
        with self._lifecycle_lock:
            if not self._is_active_project(project_id):
                return

            with self._session_factory() as session:
                svc = ProjectService(session=session)
                cfg = svc.get_pipeline_config(project_id)
            self._store_config(cfg)

            match component_type:
                case ComponentType.SOURCE:
                    source = self._component_factory.create_source(cfg.reader)
                    self._swap_component(lambda pipeline: pipeline.set_source(source, True))
                case ComponentType.PROCESSOR:
                    # Building the reference batch + downloading weights + initializing the model
                    # can take a while. Surface a "busy" flag so the UI can show a blocking overlay.
                    self._rebuild_processor(cfg, project_id)
                case ComponentType.SINK:
                    sink = self._component_factory.create_sink(cfg.writer)
                    self._swap_component(lambda pipeline: pipeline.set_sink(sink, True))
                case _ as unknown:
                    logger.error(f"Unknown component type {unknown}")

    def _rebuild_processor(self, cfg: PipelineConfig, project_id: UUID) -> None:
        """Rebuild the processor and swap it into the running pipeline."""
        token = self._new_build_token()
        try:
            processor, terminal_status = self._build_processor(cfg, project_id, token, fallback_to_passthrough=False)
            # A cancelled processor must not be swapped into a pipeline that was
            # meanwhile torn down and rebuilt.
            self._raise_if_cancelled(token)
            if self._swap_component(lambda pipeline: pipeline.set_processor(processor, True)):
                self._publish_processor_status(token, terminal_status)
        except _BuildCancelledError:
            logger.info("Discarding cancelled processor build for project %s", project_id)
        finally:
            self._finish_build(token)

    def _swap_component(self, install: Callable[[Pipeline], None]) -> bool:
        """Install a freshly built component into the active pipeline, if any.

        Returns:
            True when the component was installed.
        """
        with self._state_lock:
            if self._state is None:
                logger.info("Pipeline was deleted while building a component, discarding it")
                return False
            install(self._state.pipeline)
            return True

    def _store_config(self, cfg: PipelineConfig) -> None:
        """Record the configuration the running pipeline was last updated from."""
        with self._state_lock:
            if self._state is not None:
                self._state = replace(self._state, config=cfg)

    def _require_active_state(self, project_id: UUID) -> _ActiveState:
        """Return the active state for *project_id*, or raise.

        Snapshots the state under ``_state_lock`` so callers never observe a
        pipeline that is torn down between the check and its use.
        """
        with self._state_lock:
            state = self._state
            if state is None:
                raise PipelineNotActiveError("No active pipeline.")
            if project_id != state.pipeline.project_id:
                raise PipelineProjectMismatchError(
                    f"Project ID {project_id} does not match the active pipeline's project ID."
                )
            return state

    def get_output_slot(self, project_id: UUID) -> FrameSlot[OutputData]:
        """Get the shared output slot for reading the latest processed frame.

        External consumers (e.g. WebRTC streams) can poll this slot without
        registering or unregistering — they simply read ``slot.latest``.
        """
        return self._require_active_state(project_id).pipeline.outbound_slot

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
        pipeline = self._require_active_state(project_id).pipeline
        try:
            pipeline.seek(index)
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
        pipeline = self._require_active_state(project_id).pipeline
        try:
            return pipeline.get_frame_index()
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
        pipeline = self._require_active_state(project_id).pipeline
        try:
            return pipeline.list_frames(offset, limit)
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
        return self._require_active_state(project_id).pipeline.capture_frame()
