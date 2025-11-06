# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from db.models import PromptDB, PromptType
from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class PromptRepository(BaseRepository):
    """
    Repository responsible for low-level persistence of `PromptDB` entities.

    Responsibilities:
      - Build and execute SQLAlchemy queries.
      - Add / delete ORM entities to the session.
      - No business logic, no commits, no domain exceptions.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session)

    def add(self, prompt: PromptDB) -> None:
        """
        Add a new PromptDB entity to the session (not committed).
        """
        logger.debug(f"Adding prompt id={prompt.id} project_id={prompt.project_id}")
        self.session.add(prompt)

    def get_by_id(self, prompt_id: UUID) -> PromptDB | None:
        """
        Retrieve a prompt by primary key.
        """
        logger.debug(f"Fetching prompt by id={prompt_id}")
        stmt = select(PromptDB).where(PromptDB.id == prompt_id).options(joinedload(PromptDB.annotations))
        return self.session.scalars(stmt).unique().first()

    def get_by_id_and_project(self, prompt_id: UUID, project_id: UUID) -> PromptDB | None:
        """
        Retrieve a prompt by id constrained to a project.
        """
        logger.debug(f"Fetching prompt id={prompt_id} in project_id={project_id}")
        stmt = (
            select(PromptDB)
            .where(PromptDB.id == prompt_id, PromptDB.project_id == project_id)
            .options(joinedload(PromptDB.annotations))
        )
        return self.session.scalars(stmt).unique().first()

    def get_all_by_project(self, project_id: UUID) -> Sequence[PromptDB]:
        """
        Retrieve all prompts belonging to a project.
        """
        logger.debug(f"Fetching all prompts for project_id={project_id}")
        stmt = select(PromptDB).where(PromptDB.project_id == project_id).options(joinedload(PromptDB.annotations))
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

    def delete(self, prompt: PromptDB) -> None:
        """
        Mark a PromptDB entity for deletion (not committed).
        """
        logger.debug(f"Deleting prompt id={prompt.id} project_id={prompt.project_id}")
        self.session.delete(prompt)

    def get_paginated(self, project_id: UUID, offset: int = 0, limit: int = 10) -> tuple[Sequence[PromptDB], int]:
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
