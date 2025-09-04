# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from db.model import Base


class Label(Base):
    __tablename__ = "Label"
    name: Mapped[str] = mapped_column(nullable=False)
    color: Mapped[str] = mapped_column(nullable=False)
    pipeline_id: Mapped[UUID | None] = mapped_column(ForeignKey("Pipeline.id", ondelete="CASCADE"))
    prompt_id: Mapped[UUID | None] = mapped_column(ForeignKey("Prompt.id", ondelete="CASCADE"))
    __table_args__ = (CheckConstraint("pipeline_id IS NOT NULL OR prompt_id IS NOT NULL", name="label_parent_check"),)
