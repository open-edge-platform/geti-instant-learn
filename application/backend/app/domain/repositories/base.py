# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TypeVar, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, exists, func, select
from sqlalchemy.orm import Session

from domain.db.models import Base, ProjectDB

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository[ModelType: Base]:
    """Base repository class for database operations."""

    def __init__(self, session: Session, model: type[ModelType]) -> None:
        """Initialize the repository."""
        self.session = session
        self.model = model

    def add(self, item: ModelType) -> ModelType:
        """Add a new item to the session."""
        current_time = datetime.now(UTC)
        item.created_at, item.updated_at = current_time, current_time
        self.session.add(item)
        self.session.flush()
        logger.debug(f"Added {item}")
        return item

    def update(self, item: ModelType) -> ModelType:
        """Update an existing item in the session."""
        item.updated_at = datetime.now(UTC)
        updated = self.session.merge(item)
        self.session.flush()
        self.session.refresh(updated)
        logger.debug(f"Updated {item}")
        return updated

    def get_by_id(self, object_id: UUID) -> ModelType | None:
        """Retrieve an item by its ID."""
        return self.session.get(self.model, object_id)

    def exists(self, object_id: UUID) -> bool:
        """Check if an item exists by its ID."""
        stmt = select(exists().where(self.model.id == object_id))
        return self.session.execute(stmt).scalar() or False

    def list_all(self) -> Sequence[ModelType]:
        """List all items of the model."""
        return self.session.execute(select(self.model)).scalars().all()

    def list_with_pagination(self, offset: int = 0, limit: int = 20) -> tuple[Sequence[ModelType], int]:
        """
        List items with pagination.
        Returns:
            A tuple of (items, total_count)
        """
        total_count = self.session.scalar(select(func.count()).select_from(self.model)) or 0
        items = self.session.scalars(
            select(self.model).order_by(self.model.created_at).offset(offset).limit(limit)
        ).all()
        return items, total_count

    def delete(self, object_id: UUID) -> bool:
        """Delete an item by its ID."""
        stmt = delete(self.model).where(self.model.id == object_id)
        result = cast("CursorResult", self.session.execute(stmt))
        return result.rowcount > 0

    def add_batch(self, items: list[ModelType]) -> list[ModelType]:
        """Add a batch of items to the session."""
        current_time = datetime.now(UTC)
        for item in items:
            item.updated_at = current_time
        self.session.add_all(items)
        self.session.flush()
        return items

    def update_batch(self, updates: list[ModelType]) -> None:
        """Update a batch of items in the session."""
        current_time = datetime.now(UTC)
        for update in updates:
            update.updated_at = current_time
            self.session.merge(update)
        self.session.flush()

    def delete_batch(self, obj_ids: list[UUID]) -> int:
        """Delete a batch of items by their IDs."""
        stmt = delete(self.model).where(self.model.id.in_(obj_ids))
        result = cast("CursorResult", self.session.execute(stmt))
        return result.rowcount


class ProjectComponentRepository[ModelType: Base](BaseRepository):
    """
    Base repository for project-scoped components.
    Ensures all operations are constrained to a specific project.
    """

    def __init__(self, session: Session, model: type[ModelType]) -> None:
        """Initialize the repository with a project context."""
        super().__init__(session=session, model=model)

    def list_all_by_project(self, project_id: UUID) -> Sequence[ModelType]:
        """List all items belonging to the project."""
        stmt = select(self.model).where(self.model.project_id == project_id)
        return self.session.execute(stmt).scalars().all()

    def get_by_id_and_project(self, object_id: UUID, project_id: UUID) -> ModelType | None:
        """Retrieve an item by its ID constrained to the project."""
        stmt = select(self.model).where(self.model.id == object_id, self.model.project_id == project_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_with_pagination_by_project(
        self, project_id: UUID, offset: int = 0, limit: int = 20
    ) -> tuple[Sequence[ModelType], int]:
        """
        List items belonging to the project with pagination.
        Returns:
            A tuple of (items, total_count)
        """
        items_query = (
            select(self.model)
            .where(self.model.project_id == project_id)
            .order_by(self.model.created_at)
            .offset(offset)
            .limit(limit)
        )
        total_count_query = select(func.count()).select_from(self.model).where(self.model.project_id == project_id)

        items, total_count = (
            self.session.scalars(items_query).all(),
            self.session.scalar(total_count_query) or 0,
        )

        return items, total_count


class PipelineComponentRepository[ModelType: Base](ProjectComponentRepository):
    """
    Base repository for project-scoped components associated with the active pipelines.
    Ensures all operations are constrained to the specific active project and its pipeline.
    """

    def __init__(self, session: Session, model: type[ModelType]) -> None:
        """Initialize the repository with a project and pipeline context."""
        super().__init__(session=session, model=model)

    def get_active_in_project(self, project_id: UUID) -> ModelType | None:
        """Retrieve the active component in the active project."""
        stmt = (
            select(self.model)
            .join(ProjectDB, self.model.project_id == ProjectDB.id)
            .where(self.model.project_id == project_id, self.model.active.is_(True), ProjectDB.active.is_(True))
        )
        return self.session.execute(stmt).scalar_one_or_none()
