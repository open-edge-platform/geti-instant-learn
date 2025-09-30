# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from db.models import ProjectDB
from repositories.project import ProjectRepository
from services.common import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
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

    def __init__(self, session: Session):
        """
        Initialize the service with a SQLAlchemy session.
        """
        self.session = session
        self.project_repository = ProjectRepository(session=session)

    def create_project(self, project: ProjectDB) -> ProjectDB:
        """
        Persist and activate a new project.

        Parameters:
            project: Unpersisted ProjectDB instance (may have an explicit id).

        Raises:
            ResourceAlreadyExistsError: If name or explicit id already exists.

        Returns:
            The persisted and activated ProjectDB instance (session refreshed).
        """
        logger.debug(f"Creating project candidate name={project.name} id={(project.id if project.id else 'AUTO')}")

        if self.project_repository.exists_by_name(project.name):
            logger.error(f"Project creation rejected (duplicate name={project.name})")
            raise ResourceAlreadyExistsError(
                resource_type=ResourceType.PROJECT,
                resource_value=project.name,
                raised_by="name",
            )

        if project.id and self.project_repository.exists_by_id(project.id):
            logger.error(f"Project creation rejected (duplicate id={project.id})")
            raise ResourceAlreadyExistsError(
                resource_type=ResourceType.PROJECT,
                resource_value=str(project.id),
                raised_by="id",
            )

        self.project_repository.add(project)
        self.session.flush()  # assigns id if not provided
        self._activate_project(project)
        self.session.commit()
        self.session.refresh(project)

        logger.info(f"Created project with id={project.id}, name={project.name} (active={project.active})")
        return project

    def get_project(self, project_id: UUID) -> ProjectDB:
        """
        Retrieve a project by ID.

        Parameters:
            project_id: UUID of the project.

        Returns:
            The located project.

        Raises:
            ResourceNotFoundError: If not present.
        """
        logger.debug(f"Retrieving project id={project_id}")
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error(f"Project not found id={project_id}")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )
        return project

    def list_projects(self) -> list[ProjectDB]:
        """
        List all projects.

        Returns:
            List of `ProjectDB` entities.
        """
        logger.debug("Listing all projects")
        return list(self.project_repository.get_all())

    def update_project(self, project_id: UUID, new_name: str) -> ProjectDB:
        """
        Rename a project (enforces name uniqueness).

        Parameters:
            project_id: Target project id.
            new_name: New unique name.

        Returns:
            Updated project.

        Raises:
            ResourceNotFoundError: If project absent.
            ResourceAlreadyExistsError: If new name conflicts.
        """
        logger.debug(f"Updating project id={project_id} new_name={new_name}")
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error(f"Update failed; project not found id={project_id}")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )
        if new_name != project.name and self.project_repository.exists_by_name(new_name):
            logger.error(f"Update rejected; duplicate name={new_name} for project id={project_id}")
            raise ResourceAlreadyExistsError(
                resource_type=ResourceType.PROJECT,
                resource_value=new_name,
                raised_by="name",
            )

        project.name = new_name
        self.session.commit()
        self.session.refresh(project)
        logger.info(f"Updated project id={project.id} name={project.name}")
        return project

    def set_active_project(self, project_id: UUID) -> ProjectDB:
        """
        Mark the specified project as active (deactivates previous active project).

        Parameters:
            project_id: Project to activate.

        Returns:
            Activated project.

        Raises:
            ResourceNotFoundError: If project not found.
        """
        logger.debug(f"Activating project with id={project_id}")
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error(f"Activation failed: project with id={project_id} not found")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )
        self._activate_project(project)
        self.session.commit()
        self.session.refresh(project)
        logger.info(f"Activated project with id={project.id}")
        return project

    def get_active_project(self) -> ProjectDB:
        """
        Retrieve the active project.

        Raises:
            ResourceNotFoundError: If no active project is present.
        """
        logger.debug("Retrieving active project")
        project = self.project_repository.get_active()
        if not project:
            logger.error("No active project found")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                message="No active project found.",
            )
        return project

    def delete_project(self, project_id: UUID) -> None:
        """
        Delete a project and related non-cascaded single-relations.

        Parameters:
            project_id: Target project id.

        Raises:
            ResourceNotFoundError: If project not found.
        """
        logger.debug(f"Deleting project with id={project_id}")
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error(f"Deletion failed; project with id={project_id} not found")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )

        self.project_repository.delete(project)
        self.session.commit()
        logger.info(f"Deleted project with id={project_id}")

    def _activate_project(self, project: ProjectDB) -> None:
        """
        Ensure only one project is active.
        Deactivate the currently active project (if different) and activate the target.
        """
        current = self.project_repository.get_active()

        if current and current.id == project.id:
            return

        if current:
            logger.debug(f"Deactivating project id={current.id} before activating id={project.id}")
            current.active = False
            self.session.flush()

        project.active = True
        self.session.flush()
