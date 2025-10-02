# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from db.models.annotation import AnnotationDB
    from db.models.label import LabelDB
    from db.models.project import ProjectDB


class PromptType(str, Enum):
    """Enum for different types of prompts."""

    TEXT = "TEXT"
    VISUAL = "VISUAL"


class PromptDB(Base):
    __tablename__ = "Prompt"
    type: Mapped[PromptType] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str | None] = mapped_column(nullable=True)
    image_path: Mapped[str | None] = mapped_column(nullable=True)
    project: Mapped["ProjectDB"] = relationship(back_populates="prompts")
    annotations: Mapped[list["AnnotationDB"]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan", passive_deletes=True
    )
    labels: Mapped[list["LabelDB"]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan", passive_deletes=True
    )
