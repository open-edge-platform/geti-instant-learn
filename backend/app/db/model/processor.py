# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from uuid import UUID

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.model import Base
from db.model.project import Project


class ProcessorType(str, Enum):
    """Enum for different types of processing projects from the library."""

    # TODO update with actual processor types from Daan
    DUMMY = "DUMMY"


class Processor(Base):
    __tablename__ = "Processor"
    name: Mapped[str | None] = mapped_column(nullable=True)
    type: Mapped[ProcessorType] = mapped_column(nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    project: Mapped["Project"] = relationship(back_populates="processor", single_parent=True)
    __table_args__ = (UniqueConstraint("project_id"),)
