# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models import Base
from db.models.project import ProjectDB


class SinkDB(Base):
    __tablename__ = "Sink"
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    project: Mapped["ProjectDB"] = relationship(back_populates="sink", single_parent=True)
    __table_args__ = (UniqueConstraint("project_id"),)
