# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.error_handler import extract_constraint_name
from domain.db.constraints import CheckConstraintName, UniqueConstraintName
from domain.db.models import ProjectDB, PromptDB, PromptType
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
    ServiceError,
)
from domain.repositories.frame import FrameRepository
from domain.repositories.label import LabelRepository
from domain.repositories.project import ProjectRepository
from domain.repositories.prompt import PromptRepository
from domain.services.schemas.base import Pagination
from domain.services.schemas.mappers.prompt import (
    prompt_create_schema_to_db,
    prompt_db_to_schema,
    prompt_update_schema_to_db,
    prompts_db_to_schemas,
)
from domain.services.schemas.prompt import (
    PromptCreateSchema,
    PromptSchema,
    PromptsListSchema,
    PromptUpdateSchema,
    TextPromptCreateSchema,
    VisualPromptCreateSchema,
    VisualPromptUpdateSchema,
)

logger = logging.getLogger(__name__)


class PromptService:
    """
    Service layer orchestrating Prompt use cases.

    Responsibilities:
      - Enforce business rules.
      - Enforce invariants (single text prompt per project, visual prompts must have existing frame_id).
      - Transaction boundaries (commit).
      - Raise domain-specific exceptions.
    """

    def __init__(
        self,
        session: Session,
        prompt_repository: PromptRepository | None = None,
        project_repository: ProjectRepository | None = None,
        frame_repository: FrameRepository | None = None,
        label_repository: LabelRepository | None = None,
    ):
        """
        Initialize the service with a SQLAlchemy session.
        """
        self.session = session
        self.prompt_repository = prompt_repository or PromptRepository(session=session)
        self.project_repository = project_repository or ProjectRepository(session=session)
        self.frame_repository = frame_repository or FrameRepository()
        self.label_repository = label_repository or LabelRepository(session=session)

    def list_prompts(self, project_id: UUID, offset: int = 0, limit: int = 10) -> PromptsListSchema:
        """
        List all prompts belonging to a project with pagination.

        Parameters:
            project_id: Owning project UUID.
            offset: Number of items to skip (default: 0).
            limit: Maximum number of items to return (default: 10).

        Returns:
            Pydantic list wrapper with prompt schemas and pagination info.

        Raises:
            ResourceNotFoundError: If the project does not exist.
        """
        self._ensure_project(project_id)
        db_prompts, total_count = self.prompt_repository.get_paginated(project_id, offset=offset, limit=limit)
        prompts = prompts_db_to_schemas(db_prompts)

        pagination = Pagination(
            count=len(prompts),
            total=total_count,
            offset=offset,
            limit=limit,
        )

        return PromptsListSchema(prompts=prompts, pagination=pagination)

    def get_prompt(self, project_id: UUID, prompt_id: UUID) -> PromptSchema:
        """
        Retrieve a prompt by id within a project.

        Parameters:
            project_id: Owning project UUID.
            prompt_id: Prompt UUID.

        Raises:
            ResourceNotFoundError: If project or prompt does not exist.
        """
        self._ensure_project(project_id)
        prompt = self.prompt_repository.get_by_id_and_project(prompt_id=prompt_id, project_id=project_id)
        if not prompt:
            logger.error("Prompt not found: id=%s project_id=%s", prompt_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.PROMPT, resource_id=str(prompt_id))
        return prompt_db_to_schema(prompt)

    def create_prompt(self, project_id: UUID, create_data: PromptCreateSchema) -> PromptSchema:
        """
        Create a new prompt.
        - Visual prompts must have frame_id and annotations
        - Visual prompts must reference an existing frame
        - Annotations can optionally reference labels that must exist in the project
        - Only one text prompt is allowed per project

        Parameters:
            project_id: Owning project UUID.
            create_data: Prompt creation data.

        Returns:
            Created prompt schema.

        Raises:
            ResourceNotFoundError: If project doesn't exist or frame doesn't exist (for visual prompts).
            ResourceAlreadyExistsError: If constraint violations occur (e.g., text prompt already exists).
            ServiceError: If validation fails.
        """
        self._ensure_project(project_id)

        logger.debug(
            "Prompt create requested: project_id=%s type=%s",
            project_id,
            create_data.type,
        )

        if isinstance(create_data, TextPromptCreateSchema):
            existing_text_prompt = self.prompt_repository.get_text_prompt_by_project(project_id)
            if existing_text_prompt:
                logger.warning(
                    "Text prompt creation failed: text prompt already exists for project_id=%s (prompt_id=%s)",
                    project_id,
                    existing_text_prompt.id,
                )
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROMPT,
                    field="type",
                    message=f"A text prompt already exists for this project (ID: {existing_text_prompt.id}). "
                    "Only one text prompt is allowed per project. Please update the existing text prompt instead.",
                )

        if isinstance(create_data, VisualPromptCreateSchema):
            frame_path = self.frame_repository.get_frame_path(project_id, create_data.frame_id)
            if not frame_path:
                logger.error(
                    "Visual prompt creation failed: frame_id=%s not found in project with id=%s",
                    create_data.frame_id,
                    project_id,
                )
                raise ResourceNotFoundError(
                    resource_type=ResourceType.FRAME,
                    resource_id=str(create_data.frame_id),
                    message=f"Frame {create_data.frame_id} does not exist in project {project_id}",
                )
            self._validate_annotation_labels(create_data.annotations, project_id)

        new_prompt: PromptDB = prompt_create_schema_to_db(schema=create_data, project_id=project_id)

        self.prompt_repository.add(new_prompt)

        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            logger.error("Prompt creation failed due to constraint violation: %s", exc)
            self._handle_prompt_integrity_error(exc, new_prompt.id, project_id, new_prompt.type)

        self.session.refresh(new_prompt)
        logger.info(
            "Prompt created: prompt_id=%s project_id=%s type=%s",
            new_prompt.id,
            project_id,
            new_prompt.type,
        )
        return prompt_db_to_schema(new_prompt)

    def delete_prompt(self, project_id: UUID, prompt_id: UUID) -> None:
        """
        Delete a prompt by id within a project.

        For visual prompts:
        - Deletes the frame file from filesystem
        - Annotations are deleted automatically via cascade
        - Labels associated with the prompt are deleted via cascade

        Parameters:
            project_id: Owning project UUID.
            prompt_id: Prompt UUID.

        Raises:
            ResourceNotFoundError: If project or prompt does not exist.
        """
        self._ensure_project(project_id)
        prompt = self.prompt_repository.get_by_id_and_project(prompt_id=prompt_id, project_id=project_id)
        if not prompt:
            logger.error("Cannot delete prompt: prompt_id=%s not found in project_id=%s", prompt_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.PROMPT, resource_id=str(prompt_id))

        if prompt.type == PromptType.VISUAL and prompt.frame_id:
            frame_deleted = self.frame_repository.delete_frame(project_id, prompt.frame_id)
            if frame_deleted:
                logger.info(
                    "Deleted frame file for visual prompt: frame_id=%s project_id=%s",
                    prompt.frame_id,
                    project_id,
                )
            else:
                logger.warning(
                    "Frame file not found for deletion: frame_id=%s project_id=%s",
                    prompt.frame_id,
                    project_id,
                )

        self.prompt_repository.delete(prompt)
        self.session.commit()
        logger.info("Prompt deleted: prompt_id=%s project_id=%s", prompt_id, project_id)

    def update_prompt(self, project_id: UUID, prompt_id: UUID, update_data: PromptUpdateSchema) -> PromptSchema:
        """
        Update an existing prompt.

        For visual prompts:
        - If frame_id is updated, validates the new frame exists
        - If annotations are updated, they replace the existing ones
        - Annotation label_ids are validated to exist in the project

        Parameters:
            project_id: Owning project UUID.
            prompt_id: Prompt UUID.
            update_data: Prompt update data.

        Returns:
            Updated prompt schema.

        Raises:
            ResourceNotFoundError: If project, prompt, label, or frame doesn't exist.
            ServiceError: If validation fails.
        """
        self._ensure_project(project_id)

        prompt = self.prompt_repository.get_by_id_and_project(prompt_id=prompt_id, project_id=project_id)
        if not prompt:
            logger.error("Cannot update prompt: prompt_id=%s not found in project_id=%s", prompt_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.PROMPT, resource_id=str(prompt_id))

        if prompt.type.value != update_data.type:
            logger.error(
                "Cannot change prompt type: current=%s, requested=%s",
                prompt.type,
                update_data.type,
            )
            raise ResourceUpdateConflictError(
                resource_type=ResourceType.PROMPT,
                resource_id=str(prompt_id),
                field="type",
                message=f"Cannot change prompt type from {prompt.type} to {update_data.type}. "
                "Delete and recreate the prompt instead.",
            )

        logger.debug(
            "Prompt update requested: prompt_id=%s project_id=%s type=%s",
            prompt_id,
            project_id,
            update_data.type,
        )

        if isinstance(update_data, VisualPromptUpdateSchema):
            self._handle_visual_prompt_update(prompt, update_data, project_id)

        prompt_update_schema_to_db(prompt, update_data)

        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            logger.error("Prompt update failed due to constraint violation: %s", exc)
            self._handle_prompt_integrity_error(exc, prompt.id, project_id, prompt.type)

        self.session.refresh(prompt)
        logger.info(
            "Prompt updated: prompt_id=%s project_id=%s type=%s",
            prompt.id,
            project_id,
            prompt.type,
        )
        return prompt_db_to_schema(prompt)

    def _handle_visual_prompt_update(
        self, prompt: PromptDB, update_data: VisualPromptUpdateSchema, project_id: UUID
    ) -> None:
        """
        Handle visual prompt frame updates and cleanup.

        Args:
            prompt: The prompt being updated
            update_data: The update data containing new frame_id
            project_id: The project ID for frame validation
        """
        if update_data.frame_id is not None:
            frame_path = self.frame_repository.get_frame_path(project_id, update_data.frame_id)
            if not frame_path:
                logger.error(
                    "Visual prompt update failed: frame_id=%s not found in project_id=%s",
                    update_data.frame_id,
                    project_id,
                )
                raise ResourceNotFoundError(
                    resource_type=ResourceType.FRAME,
                    resource_id=str(update_data.frame_id),
                    message=f"Frame {update_data.frame_id} does not exist in project {project_id}",
                )

            if prompt.frame_id and prompt.frame_id != update_data.frame_id:
                old_frame_deleted = self.frame_repository.delete_frame(project_id, prompt.frame_id)
                if old_frame_deleted:
                    logger.info(
                        "Deleted old frame file: frame_id=%s project_id=%s",
                        prompt.frame_id,
                        project_id,
                    )

        if update_data.annotations is not None:
            self._validate_annotation_labels(update_data.annotations, project_id)

    def _validate_annotation_labels(self, annotations: list, project_id: UUID) -> None:
        """
        Validate that all label_ids in annotations exist in the project.

        Args:
            annotations: List of AnnotationSchema objects
            project_id: The project ID for label validation

        Raises:
            ResourceNotFoundError: If any label_id doesn't exist in the project
        """
        label_ids = {ann.label_id for ann in annotations if ann.label_id is not None}

        for label_id in label_ids:
            label = self.label_repository.get_by_id(project_id, label_id)
            if not label:
                logger.error(
                    "Label not found: label_id=%s in project_id=%s",
                    label_id,
                    project_id,
                )
                raise ResourceNotFoundError(
                    resource_type=ResourceType.LABEL,
                    resource_id=str(label_id),
                    message=f"Label {label_id} does not exist in project {project_id}",
                )

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

    def _handle_prompt_integrity_error(
        self,
        exc: IntegrityError,
        prompt_id: UUID,
        project_id: UUID,
        prompt_type: PromptType,
    ) -> None:
        """
        Handle IntegrityError with context-aware messages for prompts.

        Args:
            exc: The IntegrityError from SQLAlchemy
            prompt_id: ID of the prompt being created
            project_id: ID of the owning project
            prompt_type: Type of the prompt (TEXT or VISUAL)
        """
        error_msg = str(exc.orig).lower()
        constraint_name = extract_constraint_name(error_msg)

        logger.warning(
            "Prompt constraint violation: prompt_id=%s, project_id=%s, constraint=%s, error=%s",
            prompt_id,
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

        if ("unique" in error_msg or constraint_name) and (
            constraint_name == UniqueConstraintName.SINGLE_TEXT_PROMPT_PER_PROJECT
            or ("text" in error_msg and "type" in error_msg)
        ):
            raise ResourceAlreadyExistsError(
                resource_type=ResourceType.PROMPT,
                field="type",
                message="A text prompt already exists for this project. "
                "Please update or delete the existing text prompt.",
            )

        if "check" in error_msg or constraint_name == CheckConstraintName.PROMPT_CONTENT:
            if prompt_type == PromptType.TEXT:
                raise ServiceError("Text prompt must have non-empty text content.")
            raise ServiceError("Visual prompt must have a valid frame_id and at least one annotation.")

        logger.error(f"Unmapped constraint violation for prompt (prompt_id={prompt_id}): {error_msg}")
        raise ServiceError("Database constraint violation. Please check your input and try again.")
