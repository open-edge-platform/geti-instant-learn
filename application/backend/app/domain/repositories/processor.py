# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from domain.db.models import ProcessorDB
from domain.repositories.base import PipelineComponentRepository


class ProcessorRepository(PipelineComponentRepository[ProcessorDB]):
    """
    Repository responsible for low-level persistence of `ProcessorDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=ProcessorDB)

    def get_most_recent_by_project_and_mode(self, project_id: UUID, prompt_mode: str) -> ProcessorDB | None:
        """Return the most recently updated processor for a project and prompt_mode, or None."""
        stmt = (
            select(ProcessorDB)
            .where(ProcessorDB.project_id == project_id, ProcessorDB.prompt_mode == prompt_mode)
            .order_by(ProcessorDB.updated_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_by_project_and_mode(self, project_id: UUID, prompt_mode: str) -> Sequence[ProcessorDB]:
        """List all processors for a project and prompt_mode, ordered by most recently updated first."""
        stmt = (
            select(ProcessorDB)
            .where(ProcessorDB.project_id == project_id, ProcessorDB.prompt_mode == prompt_mode)
            .order_by(ProcessorDB.updated_at.desc())
        )
        return self.session.execute(stmt).scalars().all()

    def list_with_pagination_by_project_and_mode(
        self, project_id: UUID, prompt_mode: str, offset: int = 0, limit: int = 20
    ) -> tuple[Sequence[ProcessorDB], int]:
        """List processors for a project and prompt_mode with pagination."""
        items_query = (
            select(ProcessorDB)
            .where(ProcessorDB.project_id == project_id, ProcessorDB.prompt_mode == prompt_mode)
            .order_by(ProcessorDB.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        total_count_query = (
            select(func.count())
            .select_from(ProcessorDB)
            .where(ProcessorDB.project_id == project_id, ProcessorDB.prompt_mode == prompt_mode)
        )
        items = self.session.scalars(items_query).all()
        total_count = self.session.scalar(total_count_query) or 0
        return items, total_count
