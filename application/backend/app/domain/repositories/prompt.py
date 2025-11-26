# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from domain.db.models import PromptDB, PromptType
from domain.repositories.base import ProjectComponentRepository

logger = logging.getLogger(__name__)


class PromptRepository(ProjectComponentRepository[PromptDB]):
    """
    Repository responsible for low-level persistence of `PromptDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=PromptDB)

    def list_all_by_project(self, project_id: UUID, prompt_type: PromptType | None = None) -> Sequence[PromptDB]:
        """
        Retrieve all prompts belonging to a project with optional filtering by prompt type.
        """
        logger.debug(f"Fetching all prompts for project_id={project_id}")
        stmt = select(PromptDB).where(PromptDB.project_id == project_id).options(joinedload(PromptDB.annotations))

        if prompt_type is not None:
            stmt = stmt.where(PromptDB.type == prompt_type)

        return self.session.scalars(stmt).unique().all()

    def get_text_prompt_by_project(self, project_id: UUID) -> PromptDB | None:
        """
        Retrieve the text prompt for a project (if any).
        """
        logger.debug(f"Fetching text prompt for project_id={project_id}")
        stmt = (
            select(PromptDB)
            .where(PromptDB.project_id == project_id, PromptDB.type == PromptType.TEXT)
            .options(joinedload(PromptDB.annotations))
        )
        return self.session.scalars(stmt).unique().first()

    def list_with_pagination_by_project(
        self, project_id: UUID, offset: int = 0, limit: int = 10
    ) -> tuple[Sequence[PromptDB], int]:
        """
        Retrieve prompts with pagination.

        Returns:
            A tuple of (prompts, total_count)
        """
        logger.debug(f"Fetching prompts for project_id={project_id} with offset={offset}, limit={limit}")

        prompts_query = (
            select(PromptDB)
            .where(PromptDB.project_id == project_id)
            .options(joinedload(PromptDB.annotations))
            .offset(offset)
            .limit(limit)
        )

        total_count_query = select(func.count()).select_from(PromptDB).where(PromptDB.project_id == project_id)

        prompts = self.session.scalars(prompts_query).unique().all()
        total_count = self.session.scalar(total_count_query) or 0

        return prompts, total_count
