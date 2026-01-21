# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterable
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.error_handler import extract_constraint_name
from domain.db.constraints import UniqueConstraintName
from domain.db.models import LabelDB, ProjectDB
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
)
from domain.repositories.annotation import AnnotationRepository
from domain.repositories.label import LabelRepository
from domain.repositories.project import ProjectRepository
from domain.services.base import BaseService
from domain.services.schemas.label import (
    CategoryMappings,
    LabelCreateSchema,
    LabelSchema,
    LabelsListSchema,
    LabelUpdateSchema,
    VisualizationLabel,
)
from domain.services.schemas.mappers.label import (
    label_db_to_schema,
    label_db_to_visualization_label,
    label_schema_to_db,
    labels_db_to_list_items,
)

logger = logging.getLogger(__name__)


class LabelService(BaseService):
    """
    Service layer orchestrating label operations within a project.
    Responsibilities:
      - Enforce business rules for labels (e.g., uniqueness via DB constraints).
      - Define transaction boundaries (commit / rollback).
      - Raise domain-specific exceptions.
    """

    def __init__(
        self,
        session: Session,
        label_repository: LabelRepository | None = None,
        project_repository: ProjectRepository | None = None,
        annotation_repository: AnnotationRepository | None = None,
    ):
        """
        Initialize the service with a SQLAlchemy session.
        """
        super().__init__(session=session)
        self.label_repository = label_repository or LabelRepository(session=session)
        self.project_repository = project_repository or ProjectRepository(session=session)
        self.annotation_repository = annotation_repository or AnnotationRepository(session=session)

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
        Database constraints enforce uniqueness of label name per project.
        """
        self._ensure_project(project_id)
        logger.debug(
            "Label create requested: name=%s id=%s project_id=%s",
            create_data.name,
            create_data.id or "AUTO",
            project_id,
        )

        label: LabelDB = label_schema_to_db(create_data)
        label.project_id = project_id

        try:
            with self.db_transaction():
                self.label_repository.add(label)
        except IntegrityError as exc:
            logger.error("Label creation failed due to constraint violation: %s", exc)
            self._handle_label_integrity_error(exc, label.id, project_id, create_data.name, "create")

        self.session.refresh(label)
        logger.info(
            "Label created: id=%s name=%s project_id=%s",
            label.id,
            label.name,
            project_id,
        )
        return label_db_to_schema(label=label)

    def get_label_by_id(self, project_id: UUID, label_id: UUID) -> LabelSchema:
        """
        Retrieve a label by its ID.
        """
        self._ensure_project(project_id)
        label = self.label_repository.get_by_id_and_project(label_id, project_id)
        if not label:
            logger.error("Label not found id=%s for project_id=%s", label_id, project_id)
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
        logger.debug("Labels list for project %s requested with offset=%s, limit=%s", project_id, offset, limit)
        labels, total = self.label_repository.list_with_pagination_by_project(
            project_id=project_id, offset=offset, limit=limit
        )
        return labels_db_to_list_items(labels, total=total, offset=offset, limit=limit)

    def delete_label(self, project_id: UUID, label_id: UUID) -> None:
        """
        Delete a label by its ID if not in use.

        Raises:
            ResourceInUseError: If label is referenced by any annotations.
        """
        self._ensure_project(project_id)
        label = self.label_repository.get_by_id_and_project(label_id, project_id)
        if not label:
            logger.error("Label not found id=%s for project_id=%s", label_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.LABEL, resource_id=str(label_id))

        if self.annotation_repository.is_label_in_use(label_id):
            logger.warning(
                "Cannot delete label: label_id=%s is referenced by annotations in project_id=%s",
                label_id,
                project_id,
            )
            raise ResourceInUseError(
                resource_type=ResourceType.LABEL,
                resource_id=str(label_id),
                message=f"Label '{label.name}' cannot be deleted because it is referenced by annotations. "
                f"Please remove the prompts using this label first.",
            )

        with self.db_transaction():
            self.label_repository.delete(label_id)
        logger.info("Label deleted id=%s project_id=%s", label_id, project_id)

    def update_label(self, project_id: UUID, label_id: UUID, update_data: LabelUpdateSchema) -> LabelSchema:
        """
        Update a label:
          - Rename if `name` provided and different (enforces uniqueness via DB constraint).
          - Change color if `color` provided and different.
        """
        logger.debug(
            "Label update requested for project: %s for label id=%s name=%s",
            project_id,
            label_id,
            update_data.name,
        )
        self._ensure_project(project_id)
        label = self.label_repository.get_by_id_and_project(label_id, project_id)

        if not label:
            logger.error("Update failed; label not found id=%s in project=%s", label_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.LABEL, resource_id=str(label_id))

        try:
            with self.db_transaction():
                if update_data.name is not None and label.name != update_data.name:
                    label.name = update_data.name

                if update_data.color is not None:
                    color = update_data.color.as_hex(format="long").lower()
                    if label.color != color:
                        label.color = color

                label = self.label_repository.update(label)

        except IntegrityError as exc:
            logger.error("Label update failed due to constraint violation: %s", exc)
            self._handle_label_integrity_error(exc, label.id, project_id, update_data.name, "update")

        logger.info("Label updated in project=%s label_id=%s name=%s", project_id, label.id, label.name)
        return label_db_to_schema(label=label)

    def get_visualization_labels(self, project_id: UUID) -> list[VisualizationLabel]:
        """
        Get all project labels formatted for inference visualization.
        Converts hex color strings to RGB tuples.
        """
        labels = self.label_repository.list_all_by_project(project_id)
        return [label_db_to_visualization_label(label) for label in labels]

    def build_category_mappings(self, label_ids: Iterable[UUID]) -> CategoryMappings:
        """
        Build deterministic bidirectional category ID mappings.

        Category IDs are assigned in sorted order of label UUIDs (by string representation)
        to ensure deterministic mapping across invocations.

        """
        sorted_label_ids = sorted(set(label_ids), key=str)
        label_to_category_id = {label_id: idx for idx, label_id in enumerate(sorted_label_ids)}
        category_id_to_label_id = {idx: str(label_id) for label_id, idx in label_to_category_id.items()}

        return CategoryMappings(
            label_to_category_id=label_to_category_id,
            category_id_to_label_id=category_id_to_label_id,
        )

    def _handle_label_integrity_error(
        self,
        exc: IntegrityError,
        label_id: UUID,
        project_id: UUID,
        label_name: str | None,
        operation: str,
    ) -> None:
        """
        Handle IntegrityError with context-aware messages for labels.

        Args:
            exc: The IntegrityError from SQLAlchemy
            label_id: ID of the label being created/updated
            project_id: ID of the owning project
            label_name: Name of the label (for better error messages)
            operation: Operation being performed ("create" or "update")
        """
        error_msg = str(exc.orig).lower()
        constraint_name = extract_constraint_name(error_msg)

        logger.warning(
            "Label constraint violation during %s: label_id=%s, project_id=%s, constraint=%s, error=%s",
            operation,
            label_id,
            project_id,
            constraint_name or "unknown",
            error_msg,
        )

        if "foreign key" in error_msg:
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
                message="Referenced project does not exist.",
            )

        if "unique" in error_msg or constraint_name == UniqueConstraintName.LABEL_NAME_PER_PROJECT:
            raise ResourceAlreadyExistsError(
                resource_type=ResourceType.LABEL,
                resource_value=label_name,
                field="name",
                message=f"A label with the name '{label_name}' already exists in this project."
                if label_name
                else "A label with this name already exists in this project.",
            )

        logger.error(f"Unmapped constraint violation for label {operation} (label_id={label_id}): {error_msg}")
        raise ValueError(f"Database constraint violation during label {operation}. Please check your input.")
