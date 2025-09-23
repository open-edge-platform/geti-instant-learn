# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from db.model import Project
from db.repository.common import BaseRepository, ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType

logger = logging.getLogger(__name__)


class ProjectRepository(BaseRepository):
    def __init__(self, session: Session):
        self.session = session

    def create_project(self, name: str, project_id: UUID | None = None) -> Project:
        """
        Create a new Project in the database.

        Raises:
            ResourceAlreadyExistsError: If a project with the given name or id already exists.
        """
        filters = [Project.name == name]
        if project_id is not None:
            filters.append(Project.id == project_id)
        existing_project: Project | None = self.session.scalars(select(Project).where(or_(*filters))).first()
        if existing_project:
            if existing_project.name == name:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROJECT, resource_value=existing_project.name, raised_by="name"
                )
            raise ResourceAlreadyExistsError(
                resource_type=ResourceType.PROJECT, resource_value=str(existing_project.id), raised_by="id"
            )

        new_project = Project(name=name)
        if project_id is not None:
            new_project.id = project_id
        self.session.add(new_project)
        self.session.flush()
        self.session.refresh(new_project)
        self.set_active_project(project_id=new_project.id)
        self.session.commit()
        return new_project

    def get_active_project(self) -> Project | None:
        """
        Retrieve the currently active project.

        Returns:
            The active Project instance, or None if no project is active.

        Raises:
            RuntimeError: If more than one active project is found.
        """
        active_projects = self.session.scalars(select(Project).where(Project.active)).all()
        if len(active_projects) == 0:
            return None
        if len(active_projects) > 1:
            raise RuntimeError("More than one active project found.")
        return active_projects[0]

    def set_active_project(self, project_id: UUID) -> None:
        """
        Set the project with the given ID as active, deactivating any currently active project.

        Args:
            project_id: The UUID of the project to activate.
        """
        active_project: Project | None = self.get_active_project()
        if active_project:
            active_project.active = False

        project_to_activate: Project | None = self.session.scalars(
            select(Project).where(Project.id == project_id)
        ).first()
        if project_to_activate:
            project_to_activate.active = True
            self.session.commit()
        else:
            self.session.rollback()
            raise ValueError(f"Project with id {project_id} not found.")

    def update_project(self, project_id: UUID, new_name: str) -> Project:
        """
        Update the project with the given ID with the provided info.

        Args:
            project_id: The UUID of the project to update.
            new_name: The new name for the project.

        Returns: The updated Project instance, or None if no project with the given ID exists.
        """
        project: Project | None = self.session.scalars(select(Project).where(Project.id == project_id)).first()
        if not project:
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
        project.name = new_name
        self.session.commit()
        self.session.refresh(project)
        return project

    def get_project_by_id(self, project_id: UUID) -> Project:
        """
        Retrieve a project by its ID.

        Returns:
            The Project instance, or None if not found.
        """
        project: Project | None = self.session.scalars(select(Project).where(Project.id == project_id)).first()
        if not project:
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
        return project

    def get_all_projects(self) -> Sequence[Project]:
        """
        Retrieve all existing projects.

        Returns:
            A list of all Project instances.
        """
        return self.session.scalars(select(Project)).all()

    def delete_project(self, project_id: UUID) -> None:
        """
        Delete the project with the given ID and all related sources, processors, sinks, prompts, and labels.

        Args:
            project_id: The UUID of the project to delete.

        Raises:
            ValueError: If no project with the given ID is found.
        """
        project: Project | None = self.session.scalars(select(Project).where(Project.id == project_id)).first()
        if not project:
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        if project.source:
            self.session.delete(project.source)
        if project.processor:
            self.session.delete(project.processor)
        if project.sink:
            self.session.delete(project.sink)

        for prompt in project.prompts:
            self.session.delete(prompt)
        for label in project.labels:
            self.session.delete(label)

        self.session.delete(project)
        self.session.commit()
