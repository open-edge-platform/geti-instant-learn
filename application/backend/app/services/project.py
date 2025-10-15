# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import logging
from uuid import UUID

from sqlalchemy.orm import Session
from core.runtime.dispatcher import (
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
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
    ProjectRuntimeConfig,
    SourceSchema
)
from core.components.schemas.reader import ReaderConfig
from core.components.schemas.processor import ModelConfig
from core.components.schemas.writer import WriterConfig

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

    def __init__(self, session: Session, project_repository: ProjectRepository | None = None,
                 config_change_dispatcher: ConfigChangeDispatcher | None = None):
        """
        Initialize the service with a SQLAlchemy session.
        """
        self.session = session
        self.project_repository = project_repository or ProjectRepository(session=session)
        self._dispatcher = config_change_dispatcher

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

    def list_projects(self) -> ProjectsListSchema:
        """
        List all projects.
        """
        logger.debug("Projects list requested")
        items = projects_db_to_list_items(self.project_repository.get_all())
        return ProjectsListSchema(projects=items)

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
                self._activate_project(project)  # handles deactivation of previously active project
            else:
                logger.debug("Deactivating project id=%s via update request", project_id)
                project.active = False
                self.session.flush()
                self._emit_deactivation(project.id)
            changed = True

        if changed:
            self.session.commit()
            self.session.refresh(project)
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


    def get_project_runtime_config(self, project_id: UUID) -> ProjectRuntimeConfig:
        """
        Return full runtime configuration (project + all sources, processors, sinks).
        """
        project = self.project_repository.get_by_id(project_id)
        if not project:
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        sources: list[SourceSchema] = []
        for s in project.sources:
            try:
                reader_cfg = ReaderConfig.model_validate(s.config)
                sources.append(SourceSchema(id=s.id, connected=s.connected, config=reader_cfg))
            except Exception as exc:
                logger.error("Invalid source config skipped: source_id=%s err=%s", s.id, exc)

        processors: list[ModelConfig] = []  # TODO update later with actual processor configs
        for p in project.processors:
            try:
                processors.append(ModelConfig.model_validate(p.config))
            except Exception as exc:
                logger.error("Invalid processor config skipped: processor_id=%s err=%s", p.id, exc)

        sinks: list[WriterConfig] = []  # TODO update later with actual sink configs
        for s in project.sinks:
            try:
                sinks.append(WriterConfig.model_validate(s.config))
            except Exception as exc:
                logger.error("Invalid sink config skipped: sink_id=%s err=%s", w.id, exc)

        return ProjectRuntimeConfig(
            id=project.id,
            name=project.name,
            active=project.active,
            sources=sources,
            processors=processors,
            sinks=sinks,
        )

    def get_active_project_runtime_config(self) -> ProjectRuntimeConfig:
        """
        Convenience wrapper returning runtime config of the active project.
        """
        project = self.project_repository.get_active()
        if not project:
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                message="No active project found.",
            )
        return self.get_project_runtime_config(project.id)

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
        Emit project activation event.
        """
        if self._dispatcher:
            self._dispatcher.dispatch(ProjectActivationEvent(project_id=str(project_id)))

    def _emit_deactivation(self, project_id: UUID) -> None:
        """
        Emit project deactivation event.
        """
        if self._dispatcher:
            self._dispatcher.dispatch(ProjectDeactivationEvent(project_id=str(project_id)))
