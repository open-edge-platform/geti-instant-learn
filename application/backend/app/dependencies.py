# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from core.runtime.dispatcher import ConfigChangeDispatcher
from core.runtime.pipeline_manager import PipelineManager
from db.engine import get_session
from repositories.frame import FrameRepository
from repositories.project import ProjectRepository
from repositories.source import SourceRepository
from services import FrameService, LabelService, ProjectService, SourceService
from settings import get_settings
from webrtc.manager import WebRTCManager

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
    pipeline_manager: Annotated[PipelineManager, Depends(get_pipeline_manager)],
    frame_repo: Annotated[FrameRepository, Depends(get_frame_repository)],
    project_repo: Annotated[ProjectRepository, Depends(get_project_repository)],
    source_repo: Annotated[SourceRepository, Depends(get_source_repository)],
) -> FrameService:
    """Dependency that provides a FrameService instance."""
    return FrameService(pipeline_manager, frame_repo, project_repo, source_repo)


def get_label_service(session: SessionDep) -> LabelService:
    """Dependency that provides a LabelService instance."""
    return LabelService(session=session)


# --- Dependency aliases ---
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
SourceServiceDep = Annotated[SourceService, Depends(get_source_service)]
FrameServiceDep = Annotated[FrameService, Depends(get_frame_service)]
LabelServiceDep = Annotated[LabelService, Depends(get_label_service)]
