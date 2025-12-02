# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.error_handler import extract_constraint_name
from domain.db.constraints import UniqueConstraintName
from domain.db.models import ProjectDB, SinkDB
from domain.dispatcher import ComponentConfigChangeEvent, ConfigChangeDispatcher
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from domain.repositories.project import ProjectRepository
from domain.repositories.sink import SinkRepository
from domain.services.base import BaseService
from domain.services.schemas.mappers.sink import (
    sink_db_to_schema,
    sink_schema_to_db,
    sinks_db_to_list_items,
)
from domain.services.schemas.sink import (
    SinkCreateSchema,
    SinkSchema,
    SinksListSchema,
    SinkUpdateSchema,
)

logger = logging.getLogger(__name__)


class SinkService(BaseService):
    """
    Coordinates sink configuration workflows by:
    - enforcing domain constraints before repository operations
    - verifying projects exist prior to sink mutations
    - keeping name/type uniqueness and single-connection guarantees
    - translating integrity violations into domain-specific errors
    - emitting change events so downstream components refresh configurations
    """

    def __init__(
        self,
        session: Session,
        sink_repository: SinkRepository | None = None,
        project_repository: ProjectRepository | None = None,
        config_change_dispatcher: ConfigChangeDispatcher | None = None,
    ):
        super().__init__(session=session, config_change_dispatcher=config_change_dispatcher)
        self.sink_repository = sink_repository or SinkRepository(session=session)
        self.project_repository = project_repository or ProjectRepository(session=session)

    def list_sinks(self, project_id: UUID, offset: int = 0, limit: int = 20) -> SinksListSchema:
        """
        List sinks for the specified project with pagination.

        Args:
            project_id: UUID of the project.
            offset: Starting index of the returned items.
            limit: Maximum number of items requested.

        Returns:
            A schema containing a list of sinks with pagination metadata.

        Raises:
            ResourceNotFoundError: If the project does not exist.
        """
        self._ensure_project(project_id)
        sinks, total = self.sink_repository.list_with_pagination_by_project(
            project_id=project_id, offset=offset, limit=limit
        )
        return sinks_db_to_list_items(sinks, total, offset, limit)

    def get_sink(self, project_id: UUID, sink_id: UUID) -> SinkSchema:
        """
        Retrieve a sink by its ID within the specified project.

        Args:
            project_id: UUID of the project.
            sink_id: UUID of the sink.

        Returns:
            The sink schema.

        Raises:
            ResourceNotFoundError: If the project or sink does not exist.
        """
        self._ensure_project(project_id)
        sink = self.sink_repository.get_by_id_and_project(sink_id, project_id)
        if not sink:
            logger.error(f"Sink not found id={sink_id} project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.SINK, resource_id=str(sink_id))
        return sink_db_to_schema(sink)

    def create_sink(self, project_id: UUID, create_data: SinkCreateSchema) -> SinkSchema:
        """
        Create a new sink in the specified project.

        Args:
            project_id: UUID of the project.
            create_data: Schema containing sink creation data.

        Returns:
            The created sink schema.

        Raises:
            ResourceNotFoundError: If the project does not exist.
            ResourceAlreadyExistsError: If a sink with the same name or type exists.
        """
        self._ensure_project(project_id)

        sink_type = create_data.config.sink_type
        sink_name = create_data.config.name if hasattr(create_data.config, "name") else None

        logger.debug(
            f"Sink create requested: "
            f"project_id={project_id} "
            f"sink_type={sink_type} "
            f"name={sink_name} "
            f"active={create_data.active}"
        )
        try:
            with self.db_transaction():
                if create_data.active:
                    self._disconnect_existing_active_sink(project_id=project_id)
                new_sink: SinkDB = sink_schema_to_db(schema=create_data, project_id=project_id)
                self.sink_repository.add(new_sink)
                self._emit_component_change(project_id=project_id, sink_id=new_sink.id)
        except IntegrityError as exc:
            logger.error("Sink creation failed due to constraint violation: %s", exc)
            self._handle_sink_integrity_error(exc, new_sink.id, project_id, sink_type, sink_name)

        logger.info(
            "Sink created: "
            f"sink_id={new_sink.id} "
            f"project_id={project_id} "
            f"sink_type={new_sink.config.get('sink_type')} "
            f"active={new_sink.active} "
            f"config={new_sink.config}"
        )
        self._emit_component_change(project_id=project_id, sink_id=new_sink.id)
        return sink_db_to_schema(new_sink)

    def update_sink(
        self,
        project_id: UUID,
        sink_id: UUID,
        update_data: SinkUpdateSchema,
    ) -> SinkSchema:
        """
        Update an existing sink's configuration.

        Args:
            project_id: UUID of the project.
            sink_id: UUID of the sink.
            update_data: Schema containing sink update data.

        Returns:
            The updated sink schema.

        Raises:
            ResourceNotFoundError: If the project or sink does not exist.
            ResourceUpdateConflictError: If attempting to change sink_type.
            ResourceAlreadyExistsError: If uniqueness constraints are violated.
        """
        self._ensure_project(project_id)
        sink: SinkDB = self.sink_repository.get_by_id_and_project(sink_id, project_id)
        if not sink:
            logger.error(f"Update failed; sink not found id={sink_id} project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.SINK, resource_id=str(sink_id))

        existing_type = sink.config.get("sink_type")
        incoming_type = update_data.config.sink_type
        sink_name = update_data.config.name if hasattr(update_data.config, "name") else None

        if existing_type != incoming_type:
            logger.error(
                f"Cannot update sink: sink_type change forbidden for sink_id={sink_id} project_id={project_id} "
                f"(existing={existing_type}, incoming={incoming_type})"
            )
            raise ResourceUpdateConflictError(
                resource_type=ResourceType.SINK,
                resource_id=str(sink_id),
                field="sink_type",
            )

        try:
            with self.db_transaction():
                if update_data.active and not sink.active:
                    self._disconnect_existing_active_sink(project_id=project_id)
                sink.active = update_data.active
                sink.config = update_data.config.model_dump()
                sink = self.sink_repository.update(sink)
                self._emit_component_change(project_id=project_id, sink_id=sink.id)
        except IntegrityError as exc:
            logger.error("Sink update failed due to constraint violation: %s", exc)
            self._handle_sink_integrity_error(exc, sink.id, project_id, existing_type, sink_name)

        logger.info(
            "Sink updated: "
            f"sink_id={sink_id} "
            f"project_id={project_id} "
            f"sink_type={existing_type} "
            f"active={sink.active} "
            f"config={sink.config}"
        )
        return sink_db_to_schema(sink)

    def delete_sink(self, project_id: UUID, sink_id: UUID) -> None:
        """
        Delete a sink by its ID within the specified project.

        Args:
            project_id: UUID of the project.
            sink_id: UUID of the sink.

        Raises:
            ResourceNotFoundError: If the project or sink does not exist.
        """
        self._ensure_project(project_id)
        sink = self.sink_repository.get_by_id_and_project(sink_id, project_id)
        if not sink:
            logger.error(f"Cannot delete sink: sink_id={sink_id} not found in project_id={project_id}")
            raise ResourceNotFoundError(resource_type=ResourceType.SINK, resource_id=str(sink_id))
        with self.db_transaction():
            self.sink_repository.delete(sink.id)
            self._emit_component_change(project_id=project_id, sink_id=sink_id)
        logger.info(f"Sink deleted: sink_id={sink_id} project_id={project_id}")

    def _ensure_project(self, project_id: UUID) -> ProjectDB:
        """
        Ensure the project exists.

        Args:
            project_id: UUID of the project.

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

    def _disconnect_existing_active_sink(self, project_id: UUID) -> None:
        """
        Disconnect any currently active sink in the project.

        Args:
            project_id: UUID of the project.
        """
        active_sink = self.sink_repository.get_active_in_project(project_id)
        if active_sink:
            logger.info(f"Disconnecting previously active sink: sink_id={active_sink.id} project_id={project_id}")
            active_sink.active = False
            try:
                self.sink_repository.update(active_sink)
            except Exception:
                logger.exception("Failed to flush sink disconnection")
                raise
            self._emit_component_change(project_id=project_id, sink_id=active_sink.id)

    def _emit_component_change(self, project_id: UUID, sink_id: UUID) -> None:
        """
        Emit a component configuration change event for sinks to trigger pipeline updates.
        """
        if self._dispatcher:
            self._pending_events.append(
                ComponentConfigChangeEvent(
                    project_id=project_id,
                    component_type="sink",
                    component_id=str(sink_id),
                )
            )

    @staticmethod
    def _handle_sink_integrity_error(
        exc: IntegrityError,
        sink_id: UUID,
        project_id: UUID,
        sink_type: str | None,
        sink_name: str | None,
    ) -> None:
        """
        Handle sink-related database integrity errors.

        Args:
            exc: The IntegrityError raised by SQLAlchemy.
            sink_id: UUID of the sink.
            project_id: UUID of the project.
            sink_type: Type of the sink.
            sink_name: Name of the sink.

        Raises:
            ResourceNotFoundError: If a foreign key constraint is violated.
            ResourceAlreadyExistsError: If a uniqueness constraint is violated.
        """
        error_msg = str(exc.orig).lower()
        constraint_name = extract_constraint_name(error_msg)

        logger.warning(
            f"Sink constraint violation: sink_id={sink_id}, "
            f"project_id={project_id}, constraint={constraint_name or 'unknown'}, "
            f"error={error_msg}"
        )

        if "foreign key" in error_msg:
            raise ResourceNotFoundError(
                resource_type=ResourceType.SINK,
                resource_id=str(sink_id),
                message="Referenced project does not exist.",
            )

        if "unique" in error_msg or constraint_name:
            if constraint_name == UniqueConstraintName.SINK_NAME_PER_PROJECT or ("name" in error_msg and sink_name):
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SINK,
                    resource_value=sink_name,
                    field="name",
                    message=f"A sink with the name '{sink_name}' already exists in this project."
                    if sink_name
                    else "A sink with this name already exists in this project.",
                )
            if constraint_name == UniqueConstraintName.SINK_TYPE_PER_PROJECT or "sink_type" in error_msg:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SINK,
                    resource_value=sink_type,
                    field="sink_type",
                    message=f"A sink of type '{sink_type}' already exists in this project."
                    if sink_type
                    else "A sink of this type already exists in this project.",
                )
            if constraint_name == UniqueConstraintName.SINGLE_ACTIVE_SINK_PER_PROJECT or "active" in error_msg:
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SINK,
                    field="active",
                    message="Only one sink can be active per project at a time. "
                    "Please disconnect the current sink first.",
                )

        logger.error(f"Unmapped constraint violation for sink (sink_id={sink_id}): {error_msg}")
        raise ValueError("Database constraint violation. Please check your input and try again.")
