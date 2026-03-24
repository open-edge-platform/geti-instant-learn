# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from domain.db.engine import get_session
from domain.dispatcher import ConfigChangeDispatcher
from domain.repositories.frame import FrameRepository
from domain.repositories.processor import ProcessorRepository
from domain.repositories.project import ProjectRepository
from domain.repositories.prompt import PromptRepository
from domain.repositories.sink import SinkRepository
from domain.repositories.source import SourceRepository
from domain.services import (
    LabelService,
    ModelService,
    ProjectService,
    PromptService,
    SinkService,
    SourceService,
)
from domain.services.schemas.dataset import DatasetsListSchema
from runtime.core.components.validators.sink_connection import SinkConnectionValidator
from runtime.errors import DatasetNotFoundError
from runtime.pipeline_manager import PipelineManager
from runtime.services.frame import FrameService
from runtime.services.license import LicenseService
from runtime.services.source_type import SourceTypeService
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


def get_dataset_paths(request: Request) -> dict[UUID, Path]:
    """Dependency that provides startup-cached dataset id-to-path mapping."""
    return request.app.state.dataset_paths


def get_available_datasets(request: Request) -> DatasetsListSchema:
    """Dependency that provides startup-cached dataset metadata list."""
    available_datasets: DatasetsListSchema = request.app.state.available_datasets
    if not available_datasets.datasets:
        raise DatasetNotFoundError("No datasets found in startup cache.")
    return available_datasets


def get_dataset_path_by_id(
    dataset_id: UUID,
    dataset_paths: Annotated[dict[UUID, Path], Depends(get_dataset_paths)],
) -> Path:
    """Resolve dataset path by id from the in-memory startup cache."""
    try:
        return dataset_paths[dataset_id]
    except KeyError as exc:
        raise DatasetNotFoundError(f"Dataset with id '{dataset_id}' was not found.") from exc


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


def get_processor_repository(session: SessionDep) -> ProcessorRepository:
    """Provides a ProcessorRepository instance."""
    return ProcessorRepository(session)


def get_sink_repository(session: SessionDep) -> SinkRepository:
    """Provides a SinkRepository instance."""
    return SinkRepository(session)


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
) -> FrameService:
    """
    Dependency that provides a FrameService instance.
    """
    return FrameService(frame_repo)


def get_prompt_service(
    session: SessionDep,
    dispatcher: Annotated[ConfigChangeDispatcher, Depends(get_config_dispatcher)],
) -> PromptService:
    """Dependency that provides a PromptService instance."""
    return PromptService(
        session=session,
        config_change_dispatcher=dispatcher,
    )


def get_label_service(session: SessionDep) -> LabelService:
    """Dependency that provides a LabelService instance."""
    return LabelService(session=session)


def get_model_service(
    session: SessionDep, dispatcher: Annotated[ConfigChangeDispatcher, Depends(get_config_dispatcher)]
) -> ModelService:
    """Dependency that provides a ModelService instance."""
    return ModelService(session=session, config_change_dispatcher=dispatcher)


def get_sink_service(
    session: SessionDep,
    dispatcher: Annotated[ConfigChangeDispatcher, Depends(get_config_dispatcher)],
) -> SinkService:
    """Dependency that provides a SinkService instance."""
    return SinkService(session=session, config_change_dispatcher=dispatcher)


def get_sink_connection_validator() -> SinkConnectionValidator:
    """Dependency that provides a sink connection validator instance."""
    return SinkConnectionValidator()


def get_discovery_service() -> SourceTypeService:
    """Dependency that provides a DiscoveryService instance."""
    return SourceTypeService()


def get_license_service() -> LicenseService:
    """Dependency that provides a LicenseService instance."""
    return LicenseService()


# --- Dependency aliases ---
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
SourceServiceDep = Annotated[SourceService, Depends(get_source_service)]
FrameServiceDep = Annotated[FrameService, Depends(get_frame_service)]
LabelServiceDep = Annotated[LabelService, Depends(get_label_service)]
PromptServiceDep = Annotated[PromptService, Depends(get_prompt_service)]
PipelineManagerDep = Annotated[PipelineManager, Depends(get_pipeline_manager)]
ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
SinkServiceDep = Annotated[SinkService, Depends(get_sink_service)]
SinkConnectionValidatorDep = Annotated[SinkConnectionValidator, Depends(get_sink_connection_validator)]
DiscoveryServiceDep = Annotated[SourceTypeService, Depends(get_discovery_service)]
LicenseServiceDep = Annotated[LicenseService, Depends(get_license_service)]
DatasetPathDep = Annotated[Path, Depends(get_dataset_path_by_id)]
AvailableDatasetsDep = Annotated[DatasetsListSchema, Depends(get_available_datasets)]
