# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from db.model.annotation import Annotation
    from db.model.label import Label


class PromptType(str, Enum):
    """Enum for different types of prompts."""

    TEXT = "TEXT"
    VISUAL = "VISUAL"


class Prompt(Base):
    __tablename__ = "Prompt"
    type: Mapped[PromptType] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_id: Mapped[UUID] = mapped_column(ForeignKey("Pipeline.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str | None] = mapped_column(nullable=True)
    image_path: Mapped[str | None] = mapped_column(nullable=True)
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan", passive_deletes=True
    )
    labels: Mapped[list["Label"]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan", passive_deletes=True
    )
