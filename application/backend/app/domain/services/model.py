# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.error_handler import extract_constraint_name
from domain.db.constraints import UniqueConstraintName
from domain.db.models import ProcessorDB, ProjectDB
from domain.dispatcher import ComponentConfigChangeEvent, ConfigChangeDispatcher
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)
from domain.repositories.processor import ProcessorRepository
from domain.repositories.project import ProjectRepository
from domain.services.schemas.mappers.processor import (
    processor_db_to_schema,
    processor_schema_to_db,
    processors_db_to_list_items,
)
from domain.services.schemas.processor import (
    ProcessorCreateSchema,
    ProcessorListSchema,
    ProcessorSchema,
    ProcessorUpdateSchema,
)

logger = logging.getLogger(__name__)


class ModelConfigurationService:
    """
    Service layer orchestrating model configuration use cases.

    Responsibilities:
      - Enforce business rules.
      - Enforce invariants (
            single model configuration per project via DB constraints,
            only one active model per project via DB constraints).
      - Transaction boundaries (commit).
      - Raise domain-specific exceptions.
    """

    def __init__(
        self,
        session: Session,
        processor_repository: ProcessorRepository | None = None,
        project_repository: ProjectRepository | None = None,
        config_change_dispatcher: ConfigChangeDispatcher | None = None,
    ):
        """
        Initialize the service with a SQLAlchemy session.
        """
        self.session = session
        self.processor_repository = processor_repository or ProcessorRepository(session=session)
        self.project_repository = project_repository or ProjectRepository(session=session)
        self._dispatcher = config_change_dispatcher

    def list_model_configurations(self, project_id: UUID, offset: int = 0, limit: int = 20) -> ProcessorListSchema:
        """
        List all model configurations belonging to a project.

        Parameters:
            project_id: Owning project UUID.
            offset: Starting index (0-based)
            limit: Maximum number of items to return

        Returns:
            Pydantic list wrapper with processor schemas.

        Raises:
            ResourceNotFoundError: If the project does not exist.
        """
        self._ensure_project(project_id)
        db_model_configurations, total = self.processor_repository.get_paginated(
            project_id=project_id, offset=offset, limit=limit
        )
        return processors_db_to_list_items(db_model_configurations, total=total, offset=offset, limit=limit)

    def get_model_configuration(self, project_id: UUID, model_configuration_id: UUID) -> ProcessorSchema:
        """
        Retrieve a model configuration by id within a project.
        Parameters:
            project_id: Owning project UUID.
            model_configuration_id: Model Configuration UUID.
        Raises:
            ResourceNotFoundError: If project or model configuration does not exist.
        """
        self._ensure_project(project_id)
        model_configuration = self.processor_repository.get_by_id_and_project(
            processor_id=model_configuration_id, project_id=project_id
        )
        if not model_configuration:
            logger.error(f"Model configuration not found id={model_configuration_id} project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_configuration_id))
        return processor_db_to_schema(model_configuration)

    def create_model_configuration(self, project_id: UUID, create_data: ProcessorCreateSchema) -> ProcessorSchema:
        """
        Create a new model configuration.
        Database constraints enforce uniqueness of model configuration name per project.
        """
        self._ensure_project(project_id)

        model_type = create_data.config.model_type.value
        model_name = create_data.name

        logger.debug(
            f"Model configuration create requested: "
            f"project_id={project_id} model_type={model_type} name={model_name} active={create_data.active}"
        )

        if create_data.active:
            self._deactivate_existing_active_model(project_id=project_id)

        new_model_configuration: ProcessorDB = processor_schema_to_db(schema=create_data, project_id=project_id)
        self.processor_repository.add(new_model_configuration)

        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            logger.error("Model configuration creation failed due to constraint violation: %s", exc)
            self._handle_source_integrity_error(exc, new_model_configuration.id, project_id, model_name)

        self.session.refresh(new_model_configuration)
        logger.info(
            f"Model configuration created: "
            f"id={new_model_configuration.id} "
            f"project_id={project_id} "
            f"model_type={new_model_configuration.config.get('model_type')} "
            f"active={new_model_configuration.active} "
            f"config={new_model_configuration.config}"
        )
        self._emit_component_change(project_id=project_id, model_configuration_id=new_model_configuration.id)
        return processor_db_to_schema(new_model_configuration)

    def get_active_model_configuration(self, project_id: UUID) -> ProcessorSchema:
        """
        Retrieve the active model configuration for the project.
        Parameters:
            project_id: Owning project UUID.
        """
        self._ensure_project(project_id)
        active_model = self.processor_repository.get_activated_in_project(project_id)
        if not active_model:
            logger.error(f"No active model configuration found for project_id={project_id}")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROCESSOR,
                message="No active model configuration found for the specified project.",
            )
        logger.info(f"Active model fetched for project_id={project_id}:")
        return processor_db_to_schema(active_model)

    def update_model_configuration(
        self,
        project_id: UUID,
        model_configuration_id: UUID,
        update_data: ProcessorUpdateSchema,
    ) -> ProcessorSchema:
        """
        Update existing model configuration.
        """
        self._ensure_project(project_id)
        model_configuration = self.processor_repository.get_by_id_and_project(model_configuration_id, project_id)
        if not model_configuration:
            logger.error(
                f"Update failed; model configuration not found id={model_configuration_id} project_id={project_id}"
            )
            raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_configuration_id))

        if update_data.active and not model_configuration.active:
            self._deactivate_existing_active_model(project_id=project_id)

        # Update name if provided and different
        if update_data.name is not None and model_configuration.name != update_data.name:
            model_configuration.name = update_data.name

        model_configuration.active = update_data.active
        model_configuration.config = update_data.config.model_dump()
        model_name = update_data.name
        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            logger.error("Model configuration creation failed due to constraint violation: %s", exc)
            self._handle_source_integrity_error(exc, model_configuration.id, project_id, model_name)

        self.session.refresh(model_configuration)
        logger.info(
            f"Model configuration updated: "
            f"id={model_configuration_id} "
            f"project_id={project_id} "
            f"active={model_configuration.active} "
            f"config={model_configuration.config}"
        )
        self._emit_component_change(project_id=project_id, model_configuration_id=model_configuration_id)
        return processor_db_to_schema(model_configuration)

    def delete_model_configuration(self, project_id: UUID, model_configuration_id: UUID) -> None:
        """
        Delete a model configuration by id within a project.

        Parameters:
            project_id: Owning project UUID.
            model_configuration_id: Source UUID.

        Raises:
            ResourceNotFoundError: If project or model configuration does not exist.
        """
        self._ensure_project(project_id)
        model_configuration = self.processor_repository.get_by_id_and_project(
            processor_id=model_configuration_id, project_id=project_id
        )
        if not model_configuration:
            logger.error(
                f"Cannot delete model configuration: id={model_configuration_id} not found in project_id={project_id}"
            )
            raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_configuration_id))
        self.processor_repository.delete(model_configuration)
        self.session.commit()
        logger.info(f"Model configuration deleted: id={model_configuration_id} project_id={project_id}")
        self._emit_component_change(project_id=project_id, model_configuration_id=model_configuration_id)

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
            logger.error(f"Project not found id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
        return project

    def _deactivate_existing_active_model(self, project_id: UUID) -> None:
        """
        Deactivate any currently active model configuration in the project.
        Does not commit by itself; caller commits.
        """
        active_model = self.processor_repository.get_activated_in_project(project_id)
        if active_model:
            logger.info(f"Deactivated previously active model: id={active_model.id} project_id={project_id}")
            active_model.active = False

    def _emit_component_change(self, project_id: UUID, model_configuration_id: UUID) -> None:
        """
        Emit a component configuration change event for model configuration to trigger pipeline updates.
        """
        if self._dispatcher:
            self._dispatcher.dispatch(
                ComponentConfigChangeEvent(
                    project_id=project_id,
                    component_type="processor",
                    component_id=str(model_configuration_id),
                )
            )

    def _handle_source_integrity_error(
        self,
        exc: IntegrityError,
        model_configuration_id: UUID,
        project_id: UUID,
        model_name: str | None,
    ) -> None:
        """
        Handle IntegrityError with context-aware messages for model configuration.

        Args:
            exc: The IntegrityError from SQLAlchemy
            model_configuration_id: ID of the model configuration being created/updated
            project_id: ID of the owning project
            model_name: Name of the source (if applicable)
        """
        error_msg = str(exc.orig).lower()
        constraint_name = extract_constraint_name(error_msg)

        logger.warning(
            f"Model configuration constraint violation: "
            f"id={model_configuration_id}, "
            f"project_id={project_id}, "
            f"constraint={constraint_name or 'unknown'}, "
            f"error={error_msg}"
        )

        if "foreign key" in error_msg:
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROCESSOR,
                resource_id=str(model_configuration_id),
                message="Referenced project does not exist.",
            )

        if "unique" in error_msg or constraint_name:  #  noqa: SIM102
            if constraint_name == UniqueConstraintName.PROCESSOR_NAME_PER_PROJECT or "name" in error_msg:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROCESSOR,
                    resource_value=model_name,
                    field="name",
                    message=f"A model configuration with the name '{model_name}' already exists in this project."
                    if model_name
                    else "A model configuration with this name already exists in this project.",
                )

        logger.error(
            f"Unmapped constraint violation for model configuration (id={model_configuration_id}): {error_msg}"
        )
        raise ValueError("Database constraint violation. Please check your input and try again.")
