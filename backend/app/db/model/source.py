# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from uuid import UUID

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.model import Base
from db.model.pipeline import Pipeline


class SourceType(str, Enum):
    """Enum for different types of sources."""

    VIDEO_FILE = "VIDEO_FILE"
    WEB_CAMERA = "WEB_CAMERA"
    IMAGE_DIRECTORY = "IMAGE_DIRECTORY"


class Source(Base):
    __tablename__ = "Source"
    type: Mapped[SourceType] = mapped_column(nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    pipeline_id: Mapped[UUID] = mapped_column(ForeignKey("Pipeline.id", ondelete="CASCADE"))
    pipeline: Mapped["Pipeline"] = relationship(back_populates="source", single_parent=True)
    __table_args__ = (UniqueConstraint("pipeline_id"),)
