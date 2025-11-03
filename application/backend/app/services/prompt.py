# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.constraints import CheckConstraintName, UniqueConstraintName
from db.models import ProjectDB, PromptDB, PromptType
from exceptions.custom_errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ServiceError,
)
from exceptions.handler import extract_constraint_name
from repositories.frame import FrameRepository
from repositories.project import ProjectRepository
from repositories.prompt import PromptRepository
from services.schemas.mappers.prompt import (
    prompt_create_schema_to_db,
    prompt_db_to_schema,
    prompts_db_to_schemas,
)
from services.schemas.prompt import (
    PromptCreateSchema,
    PromptSchema,
    PromptsListSchema,
    VisualPromptCreateSchema,
)

logger = logging.getLogger(__name__)


class PromptService:
    """
    Service layer orchestrating Prompt use cases.

    Responsibilities:
      - Enforce business rules.
      - Enforce invariants (single text prompt per project, visual prompts must have frame_id).
      - Transaction boundaries (commit).
      - Raise domain-specific exceptions.
    """

    def __init__(
        self,
        session: Session,
        prompt_repository: PromptRepository | None = None,
        project_repository: ProjectRepository | None = None,
        frame_repository: FrameRepository | None = None,
    ):
        """
        Initialize the service with a SQLAlchemy session.
        """
        self.session = session
        self.prompt_repository = prompt_repository or PromptRepository(session=session)
        self.project_repository = project_repository or ProjectRepository(session=session)
        self.frame_repository = frame_repository or FrameRepository()

    def list_prompts(self, project_id: UUID) -> PromptsListSchema:
        """
        List all prompts belonging to a project.

        Parameters:
            project_id: Owning project UUID.

        Returns:
            Pydantic list wrapper with prompt schemas.

        Raises:
            ResourceNotFoundError: If the project does not exist.
        """
        self._ensure_project(project_id)
        db_prompts = self.prompt_repository.get_all_by_project(project_id)
        return PromptsListSchema(prompts=prompts_db_to_schemas(db_prompts))

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
            logger.error("Prompt not found id=%s project_id=%s", prompt_id, project_id)
            raise ResourceNotFoundError(resource_type=ResourceType.PROMPT, resource_id=str(prompt_id))
        return prompt_db_to_schema(prompt)

    def create_prompt(self, project_id: UUID, create_data: PromptCreateSchema) -> PromptSchema:
        """
        Create a new prompt.

        Database constraints enforce:
        - Uniqueness of prompt name per project
        - Single text prompt per project
        - Visual prompts must have frame_id and annotations

        Business rules:
        - Visual prompts must reference an existing frame

        Parameters:
            project_id: Owning project UUID.
            create_data: Prompt creation data.

        Returns:
            Created prompt schema.

        Raises:
            ResourceNotFoundError: If project doesn't exist or frame doesn't exist (for visual prompts).
            ResourceAlreadyExistsError: If constraint violations occur.
            ServiceError: If validation fails.
        """
        self._ensure_project(project_id)

        logger.debug(
            "Prompt create requested: project_id=%s type=%s name=%s",
            project_id,
            create_data.type,
            create_data.name,
        )

        # Validate visual prompt has existing frame
        if isinstance(create_data, VisualPromptCreateSchema):
            frame_path = self.frame_repository.get_frame_path(project_id, create_data.frame_id)
            if not frame_path:
                logger.error(
                    "Visual prompt creation failed: frame_id=%s not found in project_id=%s",
                    create_data.frame_id,
                    project_id,
                )
                raise ResourceNotFoundError(
                    resource_type=ResourceType.FRAME,
                    resource_id=str(create_data.frame_id),
                    message=f"Frame {create_data.frame_id} does not exist in project {project_id}",
                )

        new_prompt: PromptDB = prompt_create_schema_to_db(schema=create_data, project_id=project_id)
        self.prompt_repository.add(new_prompt)

        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            logger.error("Prompt creation failed due to constraint violation: %s", exc)
            self._handle_prompt_integrity_error(exc, new_prompt.id, project_id, new_prompt.type, new_prompt.name)

        self.session.refresh(new_prompt)
        logger.info(
            "Prompt created: prompt_id=%s project_id=%s type=%s name=%s",
            new_prompt.id,
            project_id,
            new_prompt.type,
            new_prompt.name,
        )
        return prompt_db_to_schema(new_prompt)

    def delete_prompt(self, project_id: UUID, prompt_id: UUID) -> None:
        """
        Delete a prompt by id within a project.

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
        self.prompt_repository.delete(prompt)
        self.session.commit()
        logger.info("Prompt deleted: prompt_id=%s project_id=%s", prompt_id, project_id)

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
        prompt_name: str,
    ) -> None:
        """
        Handle IntegrityError with context-aware messages for prompts.

        Args:
            exc: The IntegrityError from SQLAlchemy
            prompt_id: ID of the prompt being created
            project_id: ID of the owning project
            prompt_type: Type of the prompt (TEXT or VISUAL)
            prompt_name: Name of the prompt
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

        if "unique" in error_msg or constraint_name:
            if constraint_name == UniqueConstraintName.PROMPT_NAME_PER_PROJECT or "name" in error_msg:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROMPT,
                    resource_value=prompt_name,
                    field="name",
                    message=f"A prompt with the name '{prompt_name}' already exists in this project.",
                )
            if constraint_name == UniqueConstraintName.SINGLE_TEXT_PROMPT_PER_PROJECT or (
                "text" in error_msg and "type" in error_msg
            ):
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROMPT,
                    field="type",
                    message="Only one text prompt is allowed per project. "
                    "Please delete the existing text prompt before creating a new one.",
                )

        if "check" in error_msg or constraint_name in [
            CheckConstraintName.PROMPT_CONTENT,
            CheckConstraintName.VISUAL_PROMPT_FRAME,
        ]:
            if prompt_type == PromptType.TEXT:
                raise ServiceError(
                    "Text prompt must have non-empty text content.",
                )
            else:
                raise ServiceError(
                    "Visual prompt must have a valid frame_id.",
                )

        logger.error(f"Unmapped constraint violation for prompt (prompt_id={prompt_id}): {error_msg}")
        raise ValueError("Database constraint violation. Please check your input and try again.")

