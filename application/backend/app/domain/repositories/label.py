# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.db.models import AnnotationDB, LabelDB, PromptDB, PromptType
from domain.repositories.base import ProjectComponentRepository

logger = logging.getLogger(__name__)


class LabelRepository(ProjectComponentRepository[LabelDB]):
    """
    Repository responsible for low-level persistence of `LabelDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=LabelDB)

    def get_label_ids_by_project_and_prompt_type(
        self, project_id: UUID, prompt_type: PromptType | None = None
    ) -> set[UUID]:
        """
        Get all unique label IDs from prompts in a project.

        Args:
            project_id: The project UUID.
            prompt_type: Optional prompt type filter (VISUAL or TEXT).

        Returns:
            Set of unique label UUIDs referenced by annotations in the project's prompts.
        """
        stmt = (
            select(AnnotationDB.label_id)
            .join(PromptDB, AnnotationDB.prompt_id == PromptDB.id)
            .where(PromptDB.project_id == project_id)
        )

        if prompt_type is not None:
            stmt = stmt.where(PromptDB.type == prompt_type)

        stmt = stmt.distinct()
        result = self.session.scalars(stmt).all()
        return set(result)
