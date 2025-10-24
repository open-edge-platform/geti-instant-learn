# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from core.components.schemas.reader import SourceType
from core.runtime.dispatcher import ComponentConfigChangeEvent, ConfigChangeDispatcher
from db.models import ProjectDB, SourceDB
from repositories.project import ProjectRepository
from repositories.source import SourceRepository
from services.errors import (
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
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


class SourceService:
    """
    Service layer orchestrating Source configs use cases.

    Responsibilities:
      - Enforce business rules.
      - Enforce invariants (single source per type per project, immutable source_type on update).
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
        Raise an exception if a source with same source_type already exists in the project.
        """
        self._ensure_project(project_id)
        source_type_value: SourceType = create_data.config.source_type
        existing_same_type = self.source_repository.get_by_type_in_project(
            project_id=project_id, source_type=source_type_value
        )
        if existing_same_type:
            logger.error(
                "Cannot create source: project_id=%s already has source_type=%s (existing source_id=%s)",
                project_id,
                source_type_value,
                existing_same_type.id,
            )
            raise ResourceUpdateConflictError(
                resource_type=ResourceType.SOURCE,
                resource_id=str(existing_same_type.id),
                field="source_type",
                message=f"Project {project_id} already has a source of type {source_type_value}",
            )
        if create_data.connected:
            self._disconnect_existing_connected_source(project_id=project_id)
        new_source: SourceDB = source_schema_to_db(schema=create_data, project_id=project_id)
        self.source_repository.add(new_source)
        self.session.commit()
        self.session.refresh(new_source)
        logger.info(
            "Source created: source_id=%s project_id=%s source_type=%s connected=%s config=%s",
            new_source.id,
            project_id,
            source_type_value,
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

        self.session.commit()
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
        Disconnect any currently connected source in the project, except the one with exclude_id.
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
