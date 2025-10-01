# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models.prompt import PromptDB

from .base import Base


class AnnotationDB(Base):
    __tablename__ = "Annotation"
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    prompt_id: Mapped[UUID] = mapped_column(ForeignKey("Prompt.id", ondelete="CASCADE"))
    prompt: Mapped["PromptDB"] = relationship(back_populates="annotations", single_parent=True)
    __table_args__ = (UniqueConstraint("prompt_id"),)
