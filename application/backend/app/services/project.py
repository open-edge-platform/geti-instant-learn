# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from pydantic import TypeAdapter
from sqlalchemy.orm import Session

from core.components.schemas.reader import ReaderConfig
from core.runtime.dispatcher import (
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from core.runtime.schemas.pipeline import PipelineConfig
from db.models import ProjectDB
from repositories.project import ProjectRepository
from services.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)
from services.schemas.mappers.project import (
    project_db_to_schema,
    project_schema_to_db,
    projects_db_to_list_items,
)
from services.schemas.project import (
    ProjectCreateSchema,
    ProjectSchema,
    ProjectsListSchema,
    ProjectUpdateSchema,
)

logger = logging.getLogger(__name__)


class ProjectService:
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
        self.session = session
        self.project_repository = project_repository or ProjectRepository(session=session)
        self._dispatcher = config_change_dispatcher
        self._pending_events: list[ProjectActivationEvent | ProjectDeactivationEvent] = []

    def create_project(self, create_data: ProjectCreateSchema) -> ProjectSchema:
        """
        Persist and activate a new project.
        """
        logger.debug(
            "Project create requested: name=%s id=%s",
            create_data.name,
            create_data.id or "AUTO",
        )
        if self.project_repository.exists_by_name(create_data.name):
            logger.error("Project creation rejected: duplicate name=%s", create_data.name)
            raise ResourceAlreadyExistsError(
                resource_type=ResourceType.PROJECT,
                resource_value=create_data.name,
                raised_by="name",
            )

        if create_data.id and self.project_repository.exists_by_id(create_data.id):
            logger.error("Project creation rejected: duplicate id=%s", create_data.id)
            raise ResourceAlreadyExistsError(
                resource_type=ResourceType.PROJECT,
                resource_value=str(create_data.id),
                raised_by="id",
            )

        project: ProjectDB = project_schema_to_db(create_data)
        self.project_repository.add(project)
        self.session.flush()
        self._activate_project(project)
        self.session.commit()
        self.session.refresh(project)
        self._dispatch_pending_events()
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
        projects, total = self.project_repository.get_paginated(offset=offset, limit=limit)
        return projects_db_to_list_items(projects, total=total, offset=offset, limit=limit)

    def update_project(self, project_id: UUID, update_data: ProjectUpdateSchema) -> ProjectSchema:
        """
        Update a project:
          - Rename if `name` provided and different (enforces uniqueness).
          - Apply desired activation state if it differs (may result in zero active projects).
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
        changed = False

        if update_data.name is not None and update_data.name != project.name:
            if self.project_repository.exists_by_name(update_data.name):
                logger.error(
                    "Update rejected; duplicate name=%s for project id=%s",
                    update_data.name,
                    project_id,
                )
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROJECT,
                    resource_value=update_data.name,
                    raised_by="name",
                )
            logger.debug("Renaming project id=%s from '%s' to '%s'", project_id, project.name, update_data.name)
            project.name = update_data.name
            changed = True

        if update_data.active is not None and project.active != update_data.active:
            if update_data.active:
                logger.debug("Activating project id=%s via update request", project_id)
                self._activate_project(project)
            else:
                logger.debug("Deactivating project id=%s via update request", project_id)
                project.active = False
                self.session.flush()
                self._emit_deactivation(project.id)
            changed = True

        if changed:
            self.session.commit()
            self.session.refresh(project)
            self._dispatch_pending_events()
            logger.info(
                "Project updated: id=%s name=%s active=%s",
                project.id,
                project.name,
                project.active,
            )
        else:
            logger.debug("No changes applied to project id=%s", project_id)
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
        self._activate_project(project)
        self.session.commit()
        self._dispatch_pending_events()
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

        connected_source = next((s for s in project.sources if s.connected), None)
        reader_cfg: ReaderConfig | None = None
        if connected_source:
            try:
                reader_cfg = TypeAdapter(ReaderConfig).validate_python(connected_source.config)
            except Exception as exc:
                logger.exception(
                    "Invalid connected source config ignored: source_id=%s err=%s", connected_source.id, exc
                )

        return PipelineConfig(
            project_id=project.id,
            reader=reader_cfg,
            processor=None,  # TODO: populate from future processor configs
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

        if project.active:
            self._emit_deactivation(project.id)
        self.project_repository.delete(project)
        self.session.commit()
        self._dispatch_pending_events()
        logger.info("Project deleted: id=%s", project_id)

    def _activate_project(self, project: ProjectDB) -> None:
        """
        Ensure only one project is active.
        Deactivate the currently active project (if different) and activate the target.
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
            self.session.flush()
            self._emit_deactivation(current.id)

        project.active = True
        self.session.flush()
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

    def _dispatch_pending_events(self) -> None:
        """
        Dispatch and clear queued events (call only after a successful commit).
        """
        if self._dispatcher and self._pending_events:
            for ev in self._pending_events:
                self._dispatcher.dispatch(ev)
        self._pending_events.clear()
