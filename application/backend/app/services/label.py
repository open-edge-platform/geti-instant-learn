# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import secrets
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.runtime.dispatcher import (
    ConfigChangeDispatcher,
)
from db.models import LabelDB, ProjectDB
from repositories.label import LabelRepository
from repositories.project import ProjectRepository
from services.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)
from services.schemas.label import (
    LabelCreateSchema,
    LabelSchema,
    LabelsListSchema,
)
from services.schemas.mappers.label import label_db_to_schema, label_schema_to_db, labels_db_to_list_items

logger = logging.getLogger(__name__)


class LabelService:
    """
    Service layer orchestrating label operations within a project.
    Responsibilities:
      - Enforce business rules for labels (e.g., uniqueness).
      - Define transaction boundaries (commit / rollback).
      - Raise domain-specific exceptions.
    """

    def __init__(
        self,
        session: Session,
        label_repository: LabelRepository | None = None,
        project_repository: ProjectRepository | None = None,
        config_change_dispatcher: ConfigChangeDispatcher | None = None,
    ):
        """
        Initialize the service with a SQLAlchemy session.
        """
        self.session = session
        self.label_repository = label_repository or LabelRepository(session=session)
        self.project_repository = project_repository or ProjectRepository(session=session)
        self._dispatcher = config_change_dispatcher

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

    def create_label(self, project_id: UUID, create_data: LabelCreateSchema) -> LabelSchema:
        """
        Create a new label for the specified project ID.
        """
        self._ensure_project(project_id)
        logger.debug(
            "Label create requested: name=%s id=%s",
            create_data.name,
            create_data.id or "AUTO",
        )

        label: LabelDB = label_schema_to_db(create_data)
        if not label.color:
            label.color = random_color()
        label.project_id = project_id

        try:
            self.label_repository.add(label=label)
            self.session.commit()
            self.session.refresh(label)
            logger.info(
                "Label created: id=%s name=%s",
                label.id,
                label.name,
            )
            return label_db_to_schema(label=label)
        except IntegrityError as e:
            self.session.rollback()
            if "label_name_project_unique" in str(e.orig) or "name" in str(e.orig).lower():
                logger.error("Label creation rejected: duplicate name=%s", create_data.name)
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.LABEL,
                    resource_value=create_data.name,
                    raised_by="name",
                )
            if "primary key" in str(e.orig).lower() or "id" in str(e.orig).lower():
                logger.error("Label creation rejected: duplicate id=%s", create_data.id)
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.LABEL,
                    resource_value=str(create_data.id),
                    raised_by="id",
                )
            raise

    def get_label_by_id(self, project_id: UUID, label_id: UUID) -> LabelSchema:
        """
        Retrieve a label by its ID.
        """
        self._ensure_project(project_id)
        label = self.label_repository.get_by_id(project_id=project_id, label_id=label_id)
        if not label:
            logger.error(f"Label not found id={label_id} for project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.LABEL, resource_id=str(label_id))
        return label_db_to_schema(label=label)

    def get_all_labels(self, project_id: UUID, offset: int = 0, limit: int = 20) -> LabelsListSchema:
        """
        List labels with pagination.

        Parameters:
            project_id: Target project UUID
            offset: Starting index (0-based)
            limit: Maximum number of items to return

        Returns:
            LabelsListSchema with paginated results and metadata
        """
        self._ensure_project(project_id)
        logger.debug(f"Labels list for project {project_id} requested with offset={offset}, limit={limit}")
        labels, total = self.label_repository.get_paginated(project_id=project_id, offset=offset, limit=limit)
        return labels_db_to_list_items(labels, total=total, offset=offset, limit=limit)

    def delete_label(self, project_id: UUID, label_id: UUID) -> None:
        """
        Delete a label by its ID.
        """
        self._ensure_project(project_id)
        label = self.label_repository.get_by_id(project_id=project_id, label_id=label_id)
        if not label:
            logger.error(f"Label not found id={label_id} for project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.LABEL, resource_id=str(label_id))

        self.label_repository.delete(project_id=project_id, label=label)
        self.session.commit()
        logger.info("Label deleted id=%s", label_id)


def random_color() -> str:
    """
    Generate random color.
    """
    red, green, blue = (
        secrets.randbelow(256),
        secrets.randbelow(256),
        secrets.randbelow(256),
    )
    return f"#{red:02x}{green:02x}{blue:02x}"
