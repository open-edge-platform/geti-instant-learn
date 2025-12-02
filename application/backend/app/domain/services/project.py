# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from pydantic import TypeAdapter
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.error_handler import extract_constraint_name
from domain.db.constraints import UniqueConstraintName
from domain.db.models import ProjectDB
from domain.dispatcher import (
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)
from domain.repositories.project import ProjectRepository
from domain.services.base import BaseService
from domain.services.schemas.mappers.project import (
    project_db_to_schema,
    project_schema_to_db,
    projects_db_to_list_items,
)
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.processor import ModelConfig
from domain.services.schemas.project import (
    ProjectCreateSchema,
    ProjectSchema,
    ProjectsListSchema,
    ProjectUpdateSchema,
)
from domain.services.schemas.reader import ReaderConfig

logger = logging.getLogger(__name__)


class ProjectService(BaseService):
    """
    Service layer orchestrating project configs use cases.

    Responsibilities:
      - Enforce business rules (uniqueness, activation semantics).
      - Define transaction boundaries (commit / rollback).
      - Raise domain-specific exceptions.
      - Coordinate cascading / related entity cleanup.
    """

    def __init__(
        self,
        session: Session,
        project_repository: ProjectRepository | None = None,
        config_change_dispatcher: ConfigChangeDispatcher | None = None,
    ):
        """
        Initialize the service with a SQLAlchemy session.
        """
        super().__init__(session=session, config_change_dispatcher=config_change_dispatcher)
        self.project_repository = project_repository or ProjectRepository(session=session)

    def create_project(self, create_data: ProjectCreateSchema) -> ProjectSchema:
        """
        Persist and activate a new project.
        Database constraints enforce name and ID uniqueness.
        """
        logger.debug(
            "Project create requested: name=%s id=%s",
            create_data.name,
            create_data.id or "AUTO",
        )
        project: ProjectDB = project_schema_to_db(create_data)
        try:
            with self.db_transaction():
                self._activate_project(project)
                self.project_repository.add(project)
        except IntegrityError as exc:
            logger.error("Project creation failed due to constraint violation: %s", exc)
            self._handle_project_integrity_error(exc, project.id, create_data.name)

        self.session.refresh(project)
        logger.info(
            "Project created: id=%s name=%s active=%s",
            project.id,
            project.name,
            project.active,
        )
        return project_db_to_schema(project)

    def get_project(self, project_id: UUID) -> ProjectSchema:
        """
        Retrieve a project by ID.
        """
        logger.debug("Project retrieve requested: id=%s", project_id)
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error("Project not found: id=%s", project_id)
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )
        return project_db_to_schema(project)

    def list_projects(self, offset: int = 0, limit: int = 20) -> ProjectsListSchema:
        """
        List projects with pagination.

        Parameters:
            offset: Starting index (0-based)
            limit: Maximum number of items to return

        Returns:
            ProjectsListSchema with paginated results and metadata
        """
        logger.debug(f"Projects list requested with offset={offset}, limit={limit}")
        projects, total = self.project_repository.list_with_pagination(offset=offset, limit=limit)
        return projects_db_to_list_items(projects, total=total, offset=offset, limit=limit)

    def update_project(self, project_id: UUID, update_data: ProjectUpdateSchema) -> ProjectSchema:
        """
        Update a project:
          - Rename if `name` provided and different (enforces uniqueness via DB constraint).
          - Apply desired activation state if it differs.
        """
        logger.debug(
            "Project update requested: id=%s name=%s active=%s",
            project_id,
            update_data.name,
            update_data.active,
        )
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error("Update failed; project not found id=%s", project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        try:
            with self.db_transaction():
                if update_data.name is not None and update_data.name != project.name:
                    logger.debug("Renaming project id=%s from '%s' to '%s'", project_id, project.name, update_data.name)
                    project.name = update_data.name

                if update_data.active is not None and project.active != update_data.active:
                    if update_data.active:
                        logger.debug("Activating project id=%s via update request", project_id)
                        try:
                            self._activate_project(project)
                        except Exception as exc:
                            logger.error("Failed to activate project: %s", exc)
                            raise
                    else:
                        logger.debug("Deactivating project id=%s via update request", project_id)
                        project.active = False
                        self.session.flush()
                        self._emit_deactivation(project.id)

                project = self.project_repository.update(project)

        except IntegrityError as exc:
            logger.error("Project update failed due to constraint violation: %s", exc)
            self._handle_project_integrity_error(exc, project_id, update_data.name)

        logger.info(
            "Project updated: id=%s name=%s active=%s",
            project.id,
            project.name,
            project.active,
        )
        return project_db_to_schema(project)

    def set_active_project(self, project_id: UUID) -> None:
        """
        Mark the specified project as active (deactivates previous active project).

        Parameters:
            project_id: Project to activate.

        Raises:
            ResourceNotFoundError: If project not found.
        """
        logger.debug("Project activate requested: id=%s", project_id)
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error("Project activation failed: not found id=%s", project_id)
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )
        try:
            with self.db_transaction():
                self._activate_project(project)
                self.project_repository.update(project)
        except IntegrityError as exc:
            logger.error("Project update failed due to constraint violation: %s", exc)
            self._handle_project_integrity_error(exc, project_id)
        logger.info("Project activated: id=%s", project.id)

    def get_active_project_info(self) -> ProjectSchema:
        """
        Retrieve active project info.
        Raises:
            ResourceNotFoundError
        """
        logger.debug("Active project retrieve requested")
        project = self.project_repository.get_active()
        if not project:
            logger.error("Active project not found")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                message="No active project found.",
            )
        return project_db_to_schema(project)

    def get_pipeline_config(self, project_id: UUID) -> PipelineConfig:
        """
        Build and return the PipelineConfig for a specific project.

        Rules:
          - Reader: first connected source's ReaderConfig (if any), else None (NoOpReader).
          - Processor / Writer: placeholders (None) until implemented.

        Raises:
            ResourceNotFoundError: if project does not exist.
        """
        project = self.project_repository.get_by_id(project_id)
        if not project:
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        active_source = next((s for s in project.sources if s.active), None)
        reader_cfg: ReaderConfig | None = None
        if active_source:
            try:
                reader_cfg = TypeAdapter(ReaderConfig).validate_python(active_source.config)
            except Exception:
                logger.exception("Invalid connected source config ignored: source_id=%s", active_source.id)
        processor_cfg: ModelConfig | None = None
        active_model = next((m for m in project.processors if m.active), None)
        if active_model:
            try:
                processor_cfg = TypeAdapter(ModelConfig).validate_python(active_model.config)
            except Exception:
                logger.exception("Invalid active model config ignored: model_id=%s", active_model.id)

        return PipelineConfig(
            project_id=project.id,
            reader=reader_cfg,
            processor=processor_cfg,
            writer=None,  # TODO: populate from future sink/writer configs
        )

    def get_active_pipeline_config(self) -> PipelineConfig | None:
        """
        Return PipelineConfig for the active project, or None if there's no active project.
        """
        project = self.project_repository.get_active()
        if not project:
            return None
        return self.get_pipeline_config(project.id)

    def delete_project(self, project_id: UUID) -> None:
        """
        Delete a project and related non-cascaded single-relations.

        Parameters:
            project_id: Target project id.

        Raises:
            ResourceNotFoundError: If project not found.
        """
        logger.debug("Project delete requested: id=%s", project_id)
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error("Project deletion failed: not found id=%s", project_id)
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )
        with self.db_transaction():
            if project.active:
                self._emit_deactivation(project.id)
            self.project_repository.delete(project.id)
        logger.info("Project deleted: id=%s", project_id)

    def _activate_project(self, project: ProjectDB) -> None:
        """
        Ensure only one project is active.
        Deactivate the currently active project (if different) and activate the target.
        Flushes changes to DB.
        """
        current = self.project_repository.get_active()

        if current and current.id == project.id:
            return

        if current:
            logger.debug(
                "Project deactivated prior to activation: deactivated id=%s, new active id=%s",
                current.id,
                project.id,
            )
            current.active = False
            try:
                self.project_repository.update(current)
            except Exception as exc:
                logger.error("Failed to flush project deactivation: %s", exc)
                raise
            self._emit_deactivation(current.id)

        project.active = True
        self._emit_activation(project.id)

    def _emit_activation(self, project_id: UUID) -> None:
        """
        Queue project activation event (dispatched after commit).
        """
        if self._dispatcher:
            self._pending_events.append(ProjectActivationEvent(project_id=project_id))

    def _emit_deactivation(self, project_id: UUID) -> None:
        """
        Queue project deactivation event (dispatched after commit).
        """
        if self._dispatcher:
            self._pending_events.append(ProjectDeactivationEvent(project_id=project_id))

    def _handle_project_integrity_error(
        self, exc: IntegrityError, project_id: UUID, project_name: str | None = None
    ) -> None:
        """
        Handle IntegrityError with context-aware messages for projects.

        Args:
            exc: The IntegrityError from SQLAlchemy
            project_id: ID of the project being created/updated
            project_name: Name of the project (for better error messages)
        """
        error_msg = str(exc.orig).lower()
        constraint_name = extract_constraint_name(error_msg)

        logger.warning(
            "Project constraint violation: project_id=%s, constraint=%s, error=%s",
            project_id,
            constraint_name or "unknown",
            error_msg,
        )

        if "foreign key" in error_msg:
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
                message="Referenced resource does not exist.",
            )

        if "unique" in error_msg or constraint_name:
            if constraint_name == UniqueConstraintName.PROJECT_NAME or "name" in error_msg:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROJECT,
                    resource_value=project_name,
                    field="name",
                    message=f"A project with the name '{project_name}' already exists."
                    if project_name
                    else "A project with this name already exists.",
                )
            if constraint_name == UniqueConstraintName.SINGLE_ACTIVE_PROJECT or "active" in error_msg:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROJECT,
                    field="active",
                    message="Only one project can be active at a time. "
                    "Please deactivate the current active project first.",
                )

        logger.error(f"Unmapped constraint violation for project {project_id}: {error_msg}")
        raise ValueError("Database constraint violation. Please check your input and try again.")
