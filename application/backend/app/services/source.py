# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from db.models import ProjectDB
from repositories.project import ProjectRepository
from repositories.source import SourceRepository
from rest.schemas.source import (
    SourcePayloadSchema,
    SourceSchema,
    SourcesListSchema,
)
from services.common import (
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from services.mappers.source import (
    source_db_to_schema,
    source_payload_to_db,
    sources_db_to_schemas,
)

logger = logging.getLogger(__name__)


class SourceService:
    """
    Service layer orchestrating Source configs use cases.

    Responsibilities:
      - Enforce business rules.
      - Define transaction boundaries (commit / rollback).
      - Raise domain-specific exceptions.
    """

    def __init__(self, session: Session):
        """
        Initialize the service with a SQLAlchemy session.
        """
        self.session = session
        self.source_repository = SourceRepository(session=session)
        self.project_repository = ProjectRepository(session=session)

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
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
        return project

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

    def get_by_id(self, source_id: UUID) -> SourceSchema | None:
        """
        Retrieve a source by its id.

        Parameters:
            source_id: Source UUID.

        Returns:
            SourceSchema if found, else None.
        """
        db_source = self.source_repository.get_by_id(source_id)
        return source_db_to_schema(db_source) if db_source else None

    def get_by_id_and_project(self, source_id: UUID, project_id: UUID) -> SourceSchema | None:
        """
        Retrieve a source by id scoped to a project.

        Parameters:
            source_id: Source UUID.
            project_id: Project UUID.

        Returns:
            SourceSchema if found, else None.
        """
        db_source = self.source_repository.get_by_id_and_project(source_id=source_id, project_id=project_id)
        return source_db_to_schema(db_source) if db_source else None

    def upsert_source(
        self,
        project_id: UUID,
        source_id: UUID,
        payload: SourcePayloadSchema,
    ) -> tuple[SourceSchema, bool]:
        """
        Create or update a source within a project (idempotent by source_id).

        Rules:
            - If the source exists its 'type' is immutable; attempting to change it raises a conflict.
        Parameters:
            project_id: Owning project UUID.
            source_id: Desired UUID (update target or id for new entity).
            payload: Incoming configuration.
        Returns:
            (SourceSchema, created_flag) where created_flag is True if a new entity was created.
        Raises:
            ResourceNotFoundError: If the project does not exist.
            ResourceUpdateConflictError: If attempting to change an existing source's type.
        """
        self._ensure_project(project_id)
        existing_source = self.source_repository.get_by_id_and_project(source_id=source_id, project_id=project_id)
        if existing_source:
            if existing_source.type != payload.source_type:
                logger.error(
                    "Source type change forbidden source_id=%s project_id=%s existing_type=%s new_type=%s",
                    source_id,
                    project_id,
                    existing_source.type,
                    payload.source_type,
                )
                raise ResourceUpdateConflictError(
                    resource_type=ResourceType.SOURCE,
                    resource_id=str(source_id),
                    field="source_type",
                )
            logger.debug("Updating source source_id=%s, project_id=%s", source_id, project_id)
            variant_dict = payload.model_dump()
            new_config = {k: v for k, v in variant_dict.items() if k not in {"source_type", "name"}}
            existing_source.name = payload.name
            existing_source.config = new_config
            self.session.commit()
            self.session.refresh(existing_source)
            logger.info(
                "Source updated successfully, source_id=%s, project_id=%s, source_name=%s, config=%s",
                source_id,
                project_id,
                existing_source.name,
                existing_source.config,
            )
            return source_db_to_schema(existing_source), False

        logger.debug("Creating source source_id=%s, project_id=%s", source_id, project_id)
        new_source = source_payload_to_db(payload=payload, project_id=project_id)
        new_source.id = source_id
        self.source_repository.add(new_source)
        self.session.flush()
        self.session.commit()
        self.session.refresh(new_source)
        logger.info(
            "New source config created, source_id=%s, project_id=%s, source_name=%s, config=%s",
            source_id,
            project_id,
            new_source.name,
            new_source.config,
        )
        return source_db_to_schema(new_source), True

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
        source = self.source_repository.get_by_id_and_project(source_id, project_id)
        if not source:
            logger.error("Source delete failed (not found), source_id=%s project_id=%s", source_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.SOURCE, resource_id=str(source_id))
        self.source_repository.delete(source)
        self.session.commit()
        logger.info(
            "Source deleted source_id=%s project_id=%s source_name=%s",
            source_id,
            project_id,
            source.name,
        )
