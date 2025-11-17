# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from domain.db.engine import get_session
from domain.dispatcher import ConfigChangeDispatcher
from domain.repositories.frame import FrameRepository
from domain.repositories.project import ProjectRepository
from domain.repositories.prompt import PromptRepository
from domain.repositories.source import SourceRepository
from domain.services import LabelService, ProjectService, PromptService, SourceService
from runtime.pipeline_manager import PipelineManager
from runtime.services.frame import FrameService
from runtime.webrtc.manager import WebRTCManager
from settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# --- Core singletons ---
def get_pipeline_manager(request: Request) -> PipelineManager:
    """Dependency that provides access to the PipelineManager."""
    return request.app.state.pipeline_manager


def get_config_dispatcher(request: Request) -> ConfigChangeDispatcher:
    """Dependency that provides access to the ConfigChangeDispatcher."""
    return request.app.state.config_dispatcher


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager


# --- DB session dependency ---
SessionDep = Annotated[Session, Depends(get_session)]


# --- Repository providers (simple direct construction) ---
def get_project_repository(session: SessionDep) -> ProjectRepository:
    """Provides a ProjectRepository instance."""
    return ProjectRepository(session)


def get_source_repository(session: SessionDep) -> SourceRepository:
    """Provides a SourceRepository instance."""
    return SourceRepository(session)


def get_frame_repository() -> FrameRepository:
    """Provides a FrameRepository instance."""
    return FrameRepository()


def get_prompt_repository(session: SessionDep) -> PromptRepository:
    """Provides a PromptRepository instance."""
    return PromptRepository(session)


# --- Service providers ---
def get_project_service(
    session: SessionDep,
    dispatcher: Annotated[ConfigChangeDispatcher, Depends(get_config_dispatcher)],
) -> ProjectService:
    """Dependency that provides a ProjectService instance."""
    return ProjectService(session=session, config_change_dispatcher=dispatcher)


def get_source_service(
    session: SessionDep,
    dispatcher: Annotated[ConfigChangeDispatcher, Depends(get_config_dispatcher)],
) -> SourceService:
    """Dependency that provides a SourceService instance."""
    return SourceService(session=session, config_change_dispatcher=dispatcher)


def get_frame_service(
    frame_repo: Annotated[FrameRepository, Depends(get_frame_repository)],
    project_repo: Annotated[ProjectRepository, Depends(get_project_repository)],
    source_repo: Annotated[SourceRepository, Depends(get_source_repository)],
) -> FrameService:
    """
    Dependency that provides a FrameService instance without queue (for GET requests).
    This is lightweight and doesn't register any consumers with the pipeline.
    """
    return FrameService(frame_repo, project_repo, source_repo)


def get_frame_service_with_queue(
    pipeline_manager: Annotated[PipelineManager, Depends(get_pipeline_manager)],
    frame_repo: Annotated[FrameRepository, Depends(get_frame_repository)],
    project_repo: Annotated[ProjectRepository, Depends(get_project_repository)],
    source_repo: Annotated[SourceRepository, Depends(get_source_repository)],
) -> Generator[FrameService]:
    """
    Dependency that provides a FrameService instance with managed queue lifecycle (for POST requests).
    Only use this for endpoints that need to capture frames from the pipeline.
    """
    active_project = project_repo.get_active()
    if not active_project:
        # no active project - service will fail gracefully in capture_frame
        yield FrameService(frame_repo, project_repo, source_repo)
        return
    inbound_queue = pipeline_manager.register_inbound_consumer(active_project.id)

    try:
        yield FrameService(frame_repo, project_repo, source_repo, inbound_queue)
    finally:
        try:
            pipeline_manager.unregister_inbound_consumer(active_project.id, inbound_queue)
        except Exception as e:
            logger.warning(f"Failed to unregister inbound consumer queue: {e}")


def get_prompt_service(
    session: SessionDep,
    prompt_repo: Annotated[PromptRepository, Depends(get_prompt_repository)],
    project_repo: Annotated[ProjectRepository, Depends(get_project_repository)],
    frame_repo: Annotated[FrameRepository, Depends(get_frame_repository)],
) -> PromptService:
    """Dependency that provides a PromptService instance."""
    return PromptService(
        session=session,
        prompt_repository=prompt_repo,
        project_repository=project_repo,
        frame_repository=frame_repo,
    )


def get_label_service(session: SessionDep) -> LabelService:
    """Dependency that provides a LabelService instance."""
    return LabelService(session=session)


# --- Dependency aliases ---
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
SourceServiceDep = Annotated[SourceService, Depends(get_source_service)]
FrameServiceDep = Annotated[FrameService, Depends(get_frame_service)]
FrameServiceWithQueueDep = Annotated[FrameService, Depends(get_frame_service_with_queue)]
LabelServiceDep = Annotated[LabelService, Depends(get_label_service)]
PromptServiceDep = Annotated[PromptService, Depends(get_prompt_service)]
PipelineManagerDep = Annotated[PipelineManager, Depends(get_pipeline_manager)]
