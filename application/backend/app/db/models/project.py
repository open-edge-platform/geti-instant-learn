# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from sqlalchemy import Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from db.models.label import LabelDB
    from db.models.processor import ProcessorDB
    from db.models.prompt import PromptDB
    from db.models.sink import SinkDB
    from db.models.source import SourceDB


class ProjectDB(Base):
    __tablename__ = "Project"
    name: Mapped[str] = mapped_column(nullable=False)
    active: Mapped[bool] = mapped_column(nullable=False, default=False)
    source: Mapped["SourceDB"] = relationship(back_populates="project")
    processor: Mapped["ProcessorDB"] = relationship(back_populates="project")
    sink: Mapped["SinkDB"] = relationship(back_populates="project")

    prompts: Mapped[list["PromptDB"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    labels: Mapped[list["LabelDB"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    __table_args__ = (
        # ensures at most one row where active is true
        Index(
            "single_active_project",
            "active",
            unique=True,
            sqlite_where=active.is_(True),
        ),
    )
