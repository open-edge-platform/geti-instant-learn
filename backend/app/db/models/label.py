# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models import Base

if TYPE_CHECKING:
    from db.models.project import ProjectDB
    from db.models.prompt import PromptDB


class LabelDB(Base):
    __tablename__ = "Label"
    name: Mapped[str] = mapped_column(nullable=False)
    color: Mapped[str] = mapped_column(nullable=False)
    project_id: Mapped[UUID | None] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    prompt_id: Mapped[UUID | None] = mapped_column(ForeignKey("Prompt.id", ondelete="CASCADE"))
    prompt: Mapped["PromptDB"] = relationship(back_populates="labels")
    project: Mapped["ProjectDB"] = relationship(back_populates="labels")
    __table_args__ = (CheckConstraint("project_id IS NOT NULL OR prompt_id IS NOT NULL", name="label_parent_check"),)
