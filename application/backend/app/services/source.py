# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.components.schemas.reader import FrameListResponse, FrameMetadata
from core.runtime.dispatcher import ComponentConfigChangeEvent, ConfigChangeDispatcher
from db.constraints import UniqueConstraintName
from db.models import ProjectDB, SourceDB
from exceptions.custom_errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from exceptions.handler import extract_constraint_name
from repositories.project import ProjectRepository
from repositories.source import SourceRepository
from services.schemas.mappers.source import (
    source_db_to_schema,
    source_schema_to_db,
    sources_db_to_schemas,
)
from services.schemas.source import (
    SourceCreateSchema,
    SourceSchema,
    SourcesListSchema,
    SourceUpdateSchema,
)

logger = logging.getLogger(__name__)

MOCK_FILE = os.getenv("MOCKED_FILE", os.path.join(os.path.dirname(__file__), "../../../ui/src/assets/test.webp"))


class SourceService:
    """
    Service layer orchestrating Source configs use cases.

    Responsibilities:
      - Enforce business rules.
      - Enforce invariants (single source per type per project via DB constraints, immutable source_type on update).
      - Transaction boundaries (commit).
      - Raise domain-specific exceptions.
    """

    def __init__(
        self,
        session: Session,
        source_repository: SourceRepository | None = None,
        project_repository: ProjectRepository | None = None,
        config_change_dispatcher: ConfigChangeDispatcher | None = None,
    ):
        """
        Initialize the service with a SQLAlchemy session.
        """
        self.session = session
        self.source_repository = source_repository or SourceRepository(session=session)
        self.project_repository = project_repository or ProjectRepository(session=session)
        self._dispatcher = config_change_dispatcher

    def list_sources(self, project_id: UUID) -> SourcesListSchema:
        """
        List all sources belonging to a project.

        Parameters:
            project_id: Owning project UUID.

        Returns:
            Pydantic list wrapper with source schemas.

        Raises:
            ResourceNotFoundError: If the project does not exist.
        """
        self._ensure_project(project_id)
        db_sources = self.source_repository.get_all_by_project(project_id)
        return SourcesListSchema(sources=sources_db_to_schemas(db_sources))

    def get_source(self, project_id: UUID, source_id: UUID) -> SourceSchema:
        """
        Retrieve a source by id within a project.
        Parameters:
            project_id: Owning project UUID.
            source_id: Source UUID.
        Raises:
            ResourceNotFoundError: If project or source does not exist.
        """
        self._ensure_project(project_id)
        source = self.source_repository.get_by_id_and_project(source_id=source_id, project_id=project_id)
        if not source:
            logger.error("Source not found id=%s project_id=%s", source_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.SOURCE, resource_id=str(source_id))
        return source_db_to_schema(source)

    def create_source(self, project_id: UUID, create_data: SourceCreateSchema) -> SourceSchema:
        """
        Create a new source.
        Database constraints enforce uniqueness of source_type and name per project.
        """
        self._ensure_project(project_id)

        source_type = create_data.config.source_type.value
        source_name = create_data.config.name if hasattr(create_data.config, "name") else None

        logger.debug(
            "Source create requested: project_id=%s source_type=%s name=%s connected=%s",
            project_id,
            source_type,
            source_name,
            create_data.connected,
        )

        if create_data.connected:
            self._disconnect_existing_connected_source(project_id=project_id)

        new_source: SourceDB = source_schema_to_db(schema=create_data, project_id=project_id)
        self.source_repository.add(new_source)

        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            logger.error("Source creation failed due to constraint violation: %s", exc)
            self._handle_source_integrity_error(exc, new_source.id, project_id, source_type, source_name)

        self.session.refresh(new_source)
        logger.info(
            "Source created: source_id=%s project_id=%s source_type=%s connected=%s config=%s",
            new_source.id,
            project_id,
            new_source.config.get("source_type"),
            new_source.connected,
            new_source.config,
        )
        self._emit_component_change(project_id=project_id, source_id=new_source.id)
        return source_db_to_schema(new_source)

    def update_source(
        self,
        project_id: UUID,
        source_id: UUID,
        update_data: SourceUpdateSchema,
    ) -> SourceSchema:
        """
        Update existing source config (cannot change source_type).
        """
        self._ensure_project(project_id)
        source = self.source_repository.get_by_id_and_project(source_id, project_id)
        if not source:
            logger.error("Update failed; source not found id=%s project_id=%s", source_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.SOURCE, resource_id=str(source_id))

        existing_type = source.config.get("source_type")
        incoming_type = update_data.config.source_type.value
        source_name = update_data.config.name if hasattr(update_data.config, "name") else None

        if existing_type != incoming_type:
            logger.error(
                "Cannot update source: source_type change forbidden for source_id=%s project_id=%s "
                "(existing=%s, incoming=%s)",
                source_id,
                project_id,
                existing_type,
                incoming_type,
            )
            raise ResourceUpdateConflictError(
                resource_type=ResourceType.SOURCE,
                resource_id=str(source_id),
                field="source_type",
            )
        if update_data.connected and not source.connected:
            self._disconnect_existing_connected_source(project_id=project_id)

        source.connected = update_data.connected
        source.config = update_data.config.model_dump()

        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            logger.error("Source update failed due to constraint violation: %s", exc)
            self._handle_source_integrity_error(exc, source.id, project_id, existing_type, source_name)

        self.session.refresh(source)
        logger.info(
            "Source updated: source_id=%s project_id=%s source_type=%s connected=%s config=%s",
            source_id,
            project_id,
            existing_type,
            source.connected,
            source.config,
        )
        self._emit_component_change(project_id=project_id, source_id=source.id)
        return source_db_to_schema(source)

    def delete_source(self, project_id: UUID, source_id: UUID) -> None:
        """
        Delete a source by id within a project.

        Parameters:
            project_id: Owning project UUID.
            source_id: Source UUID.

        Raises:
            ResourceNotFoundError: If project or source does not exist.
        """
        self._ensure_project(project_id)
        source = self.source_repository.get_by_id_and_project(source_id=source_id, project_id=project_id)
        if not source:
            logger.error("Cannot delete source: source_id=%s not found in project_id=%s", source_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.SOURCE, resource_id=str(source_id))
        self.source_repository.delete(source)
        self.session.commit()
        logger.info("Source deleted: source_id=%s project_id=%s", source_id, project_id)
        self._emit_component_change(project_id=project_id, source_id=source_id)

    def _ensure_project(self, project_id: UUID) -> ProjectDB:
        """
        Ensure the project exists.

        Parameters:
            project_id: Target project UUID.

        Returns:
            The ProjectDB entity.

        Raises:
            ResourceNotFoundError: If the project does not exist.
        """
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error("Project not found id=%s", project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
        return project

    def _disconnect_existing_connected_source(self, project_id: UUID) -> None:
        """
        Disconnect any currently connected source in the project.
        Does not commit by itself; caller commits.
        """
        connected_source = self.source_repository.get_connected_in_project(project_id)
        if connected_source:
            logger.info(
                "Disconnecting previously connected source: source_id=%s project_id=%s",
                connected_source.id,
                project_id,
            )
            connected_source.connected = False

    def _emit_component_change(self, project_id: UUID, source_id: UUID) -> None:
        """
        Emit a component configuration change event for sources to trigger pipeline updates.
        """
        if self._dispatcher:
            self._dispatcher.dispatch(
                ComponentConfigChangeEvent(
                    project_id=project_id,
                    component_type="source",
                    component_id=str(source_id),
                )
            )

    def _handle_source_integrity_error(
        self,
        exc: IntegrityError,
        source_id: UUID,
        project_id: UUID,
        source_type: str | None,
        source_name: str | None,
    ) -> None:
        """
        Handle IntegrityError with context-aware messages for sources.

        Args:
            exc: The IntegrityError from SQLAlchemy
            source_id: ID of the source being created/updated
            project_id: ID of the owning project
            source_type: Type of the source (e.g., "GETI", "FILE")
            source_name: Name of the source (if applicable)
        """
        error_msg = str(exc.orig).lower()
        constraint_name = extract_constraint_name(error_msg)

        logger.warning(
            "Source constraint violation: source_id=%s, project_id=%s, constraint=%s, error=%s",
            source_id,
            project_id,
            constraint_name or "unknown",
            error_msg,
        )

        if "foreign key" in error_msg:
            raise ResourceNotFoundError(
                resource_type=ResourceType.SOURCE,
                resource_id=str(source_id),
                message="Referenced project does not exist.",
            )

        if "unique" in error_msg or constraint_name:
            if constraint_name == UniqueConstraintName.SOURCE_NAME_PER_PROJECT or ("name" in error_msg and source_name):
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SOURCE,
                    resource_value=source_name,
                    field="name",
                    message=f"A source with the name '{source_name}' already exists in this project."
                    if source_name
                    else "A source with this name already exists in this project.",
                )
            if constraint_name == UniqueConstraintName.SOURCE_TYPE_PER_PROJECT or "source_type" in error_msg:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SOURCE,
                    resource_value=source_type,
                    field="source_type",
                    message=f"A source of type '{source_type}' already exists in this project."
                    if source_type
                    else "A source of this type already exists in this project.",
                )
            if constraint_name == UniqueConstraintName.SINGLE_CONNECTED_SOURCE_PER_PROJECT or "connected" in error_msg:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SOURCE,
                    field="connected",
                    message="Only one source can be connected per project at a time. "
                    "Please disconnect the current source first.",
                )

        logger.error(f"Unmapped constraint violation for source (source_id={source_id}): {error_msg}")
        raise ValueError("Database constraint violation. Please check your input and try again.")

    def get_frames(self, project_id: UUID, source_id: UUID) -> FrameListResponse:
        """
        Retrieve frames from a source by id within a project.
        Parameters:
            project_id: Owning project UUID.
            source_id: Source UUID.
        """
        self._ensure_project(project_id)
        source = self.source_repository.get_by_id_and_project(source_id=source_id, project_id=project_id)
        if not source:
            logger.error("Source not found id=%s project_id=%s", source_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.SOURCE, resource_id=str(source_id))
        # Placeholder for actual frame retrieval logic
        logger.info("Retrieving frames from source_id=%s project_id=%s", source_id, project_id)
        frames = []
        with open(MOCK_FILE, "rb") as file:
            encoded_file = base64.b64encode(file.read()).decode("utf-8")
        for i in range(4):
            frames.append(FrameMetadata(index=i, thumbnail=encoded_file))
        return FrameListResponse(
            frames=frames,
            total=len(frames),
            page=1,
            page_size=len(frames),
        )
