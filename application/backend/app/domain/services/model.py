# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.error_handler import extract_constraint_name
from domain.db.constraints import UniqueConstraintName
from domain.db.models import ProcessorDB, ProjectDB
from domain.dispatcher import ComponentConfigChangeEvent, ComponentType, ConfigChangeDispatcher
from domain.errors import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from domain.repositories.processor import ProcessorRepository
from domain.repositories.project import ProjectRepository
from domain.services.base import BaseService
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


class ModelService(BaseService):
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
        super().__init__(session=session, config_change_dispatcher=config_change_dispatcher)
        self.processor_repository = processor_repository or ProcessorRepository(session=session)
        self.project_repository = project_repository or ProjectRepository(session=session)

    def list_models(self, project_id: UUID, offset: int = 0, limit: int = 20) -> ProcessorListSchema:
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
        db_models, total = self.processor_repository.list_with_pagination_by_project(
            project_id=project_id, offset=offset, limit=limit
        )
        return processors_db_to_list_items(db_models, total=total, offset=offset, limit=limit)

    def get_model(self, project_id: UUID, model_id: UUID) -> ProcessorSchema:
        """
        Retrieve a model configuration by id within a project.
        Parameters:
            project_id: Owning project UUID.
            model_id: Model Configuration UUID.
        Raises:
            ResourceNotFoundError: If project or model configuration does not exist.
        """
        self._ensure_project(project_id)
        model = self.processor_repository.get_by_id_and_project(model_id, project_id)
        if not model:
            logger.error(f"Model configuration not found id={model_id} project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_id))
        return processor_db_to_schema(model)

    def create_model(self, project_id: UUID, create_data: ProcessorCreateSchema) -> ProcessorSchema:
        """
        Create a new model configuration.
        Database constraints enforce uniqueness of model configuration name per project.
        """
        self._ensure_project(project_id)
        logger.debug(
            f"Model configuration create requested: project_id={project_id} "
            f"model_type={create_data.config.model_type.value} name={create_data.name} active={create_data.active}"
        )
        try:
            with self.db_transaction():
                if create_data.active:
                    self._deactivate_existing_active_model(project_id=project_id)
                new_model: ProcessorDB = processor_schema_to_db(schema=create_data, project_id=project_id)
                self.processor_repository.add(new_model)
                self._emit_component_change(project_id=project_id, model_id=new_model.id)
        except IntegrityError as exc:
            logger.error("Model configuration creation failed due to constraint violation: %s", exc)
            self._handle_source_integrity_error(exc, new_model.id, project_id, create_data.name)

        self.session.refresh(new_model)
        logger.info(
            f"Model configuration created: "
            f"id={new_model.id} "
            f"project_id={project_id} "
            f"model_type={new_model.config.get('model_type')} "
            f"active={new_model.active} "
            f"config={new_model.config}"
        )

        return processor_db_to_schema(new_model)

    def get_active_model(self, project_id: UUID) -> ProcessorSchema:
        """
        Retrieve the active model configuration for the project.
        Parameters:
            project_id: Owning project UUID.
        """
        self._ensure_project(project_id)
        active_model = self.processor_repository.get_active_in_project(project_id)
        if not active_model:
            logger.error(f"No active model configuration found for project_id={project_id}")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROCESSOR,
                message="No active model configuration found for the specified project.",
            )
        logger.info(f"Active model fetched for project_id={project_id}:")
        return processor_db_to_schema(active_model)

    def update_model(
        self,
        project_id: UUID,
        model_id: UUID,
        update_data: ProcessorUpdateSchema,
    ) -> ProcessorSchema:
        """
        Update existing model configuration.
        """
        self._ensure_project(project_id)
        model = self.processor_repository.get_by_id_and_project(model_id, project_id)
        if not model:
            logger.error(f"Update failed; model configuration not found id={model_id} project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_id))

        try:
            with self.db_transaction():
                if update_data.active:
                    self._deactivate_existing_active_model(project_id=project_id)
                if update_data.name is not None and model.name != update_data.name:
                    model.name = update_data.name
                model.active = update_data.active
                model.config = update_data.config.model_dump()
                model = self.processor_repository.update(model)
                self._emit_component_change(project_id=project_id, model_id=model_id)
        except IntegrityError as exc:
            logger.error("Model configuration update failed due to constraint violation: %s", exc)
            self._handle_source_integrity_error(exc, model.id, project_id, update_data.name)

        logger.info(
            f"Model configuration updated: id={model_id} project_id={project_id} active={model.active} "
            f"config={model.config}"
        )
        return processor_db_to_schema(model)

    def delete_model(self, project_id: UUID, model_id: UUID) -> None:
        """
        Delete a model configuration by id within a project.

        Parameters:
            project_id: Owning project UUID.
            model_id: Source UUID.

        Raises:
            ResourceNotFoundError: If project or model configuration does not exist.
        """
        self._ensure_project(project_id)
        model = self.processor_repository.get_by_id_and_project(model_id, project_id)
        if not model:
            logger.error(f"Cannot delete model: id={model_id} not found in project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_id))

        with self.db_transaction():
            self.processor_repository.delete(model.id)
            self._emit_component_change(project_id=project_id, model_id=model_id)
        logger.info(f"Model deleted: id={model_id} project_id={project_id}")

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
        Flushes changes to DB and emits deactivation event.
        Does not commit by itself; caller commits.
        """
        active_model = self.processor_repository.get_active_in_project(project_id)
        if active_model:
            logger.info(f"Deactivated previously active model: id={active_model.id} project_id={project_id}")
            active_model.active = False
            try:
                active_model = self.processor_repository.update(active_model)
            except Exception:
                logger.exception(f"Failed to flush deactivation of model id={active_model.id}, project_id={project_id}")
                raise
            self._emit_component_change(project_id=project_id, model_id=active_model.id)

    def _emit_component_change(self, project_id: UUID, model_id: UUID) -> None:
        """
        Emit a component configuration change event for model to trigger pipeline updates.
        """
        if self._dispatcher:
            self._pending_events.append(
                ComponentConfigChangeEvent(
                    project_id=project_id, component_type=ComponentType.PROCESSOR, component_id=model_id
                )
            )

    @staticmethod
    def _handle_source_integrity_error(
        exc: IntegrityError,
        model_id: UUID,
        project_id: UUID,
        model_name: str | None,
    ) -> None:
        """
        Handle IntegrityError with context-aware messages for model configuration.

        Args:
            exc: The IntegrityError from SQLAlchemy
            model_id: ID of the model configuration being created/updated
            project_id: ID of the owning project
            model_name: Name of the source (if applicable)
        """
        error_msg = str(exc.orig).lower()
        constraint_name = extract_constraint_name(error_msg)

        logger.warning(
            f"Model constraint violation: "
            f"id={model_id}, "
            f"project_id={project_id}, "
            f"constraint={constraint_name or 'unknown'}, "
            f"error={error_msg}"
        )

        if "foreign key" in error_msg:
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROCESSOR,
                resource_id=str(model_id),
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

        logger.error(f"Unmapped constraint violation for model configuration (id={model_id}): {error_msg}")
        raise ValueError("Database constraint violation. Please check your input and try again.")
