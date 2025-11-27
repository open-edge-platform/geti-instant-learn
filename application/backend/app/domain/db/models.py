# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Text, UniqueConstraint
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from domain.db.constraints import CheckConstraintName, UniqueConstraintName


class Base(DeclarativeBase):
    __abstract__ = True
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())


class LabelDB(Base):
    __tablename__ = "Label"
    name: Mapped[str] = mapped_column(nullable=False)
    color: Mapped[str] = mapped_column(nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"), nullable=False)
    project: Mapped["ProjectDB"] = relationship(back_populates="labels")
    __table_args__ = (UniqueConstraint("name", "project_id", name=UniqueConstraintName.LABEL_NAME_PER_PROJECT),)


class AnnotationDB(Base):
    __tablename__ = "Annotation"
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    label_id: Mapped[UUID | None] = mapped_column(ForeignKey("Label.id", ondelete="SET NULL"), nullable=True)
    prompt_id: Mapped[UUID] = mapped_column(ForeignKey("Prompt.id", ondelete="CASCADE"))
    prompt: Mapped["PromptDB"] = relationship(back_populates="annotations", single_parent=True)


class SourceDB(Base):
    __tablename__ = "Source"
    active: Mapped[bool] = mapped_column(nullable=False, default=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    project: Mapped["ProjectDB"] = relationship(back_populates="sources")
    __table_args__ = (
        Index(
            UniqueConstraintName.SOURCE_TYPE_PER_PROJECT,
            "project_id",
            sa_text("json_extract(config, '$.source_type')"),
            unique=True,
        ),
        Index(
            UniqueConstraintName.SOURCE_NAME_PER_PROJECT,
            "project_id",
            sa_text("json_extract(config, '$.name')"),
            unique=True,
            sqlite_where=sa_text("json_extract(config, '$.name') IS NOT NULL"),
        ),
        Index(
            UniqueConstraintName.SINGLE_CONNECTED_SOURCE_PER_PROJECT,
            "project_id",
            "active",
            unique=True,
            sqlite_where=sa_text("active IS 1"),
        ),
    )


class SinkDB(Base):
    __tablename__ = "Sink"
    active: Mapped[bool] = mapped_column(nullable=False, default=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    project: Mapped["ProjectDB"] = relationship(back_populates="sinks", single_parent=True)
    __table_args__ = (
        Index(
            UniqueConstraintName.SINK_TYPE_PER_PROJECT,
            "project_id",
            sa_text("json_extract(config, '$.sink_type')"),
            unique=True,
        ),
        Index(
            UniqueConstraintName.SINK_NAME_PER_PROJECT,
            "project_id",
            sa_text("json_extract(config, '$.name')"),
            unique=True,
            sqlite_where=sa_text("json_extract(config, '$.name') IS NOT NULL"),
        ),
        Index(
            UniqueConstraintName.SINGLE_ACTIVE_SINK_PER_PROJECT,
            "project_id",
            "active",
            unique=True,
            sqlite_where=sa_text("active IS 1"),
        ),
    )


class PromptType(StrEnum):
    """Enum for different types of prompts."""

    TEXT = "TEXT"
    VISUAL = "VISUAL"


class PromptDB(Base):
    __tablename__ = "Prompt"
    type: Mapped[PromptType] = mapped_column(nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    frame_id: Mapped[UUID | None] = mapped_column(nullable=True)
    thumbnail: Mapped[str | None] = mapped_column(Text, nullable=True)  # base64-encoded image with annotations
    project: Mapped["ProjectDB"] = relationship(back_populates="prompts")
    annotations: Mapped[list[AnnotationDB]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan", passive_deletes=True
    )
    __table_args__ = (
        # ensure only one text prompt per project
        Index(
            UniqueConstraintName.SINGLE_TEXT_PROMPT_PER_PROJECT,
            "project_id",
            "type",
            unique=True,
            sqlite_where=sa_text("type = 'TEXT'"),
        ),
        # ensure each frame can only be used once across all prompts
        Index(
            UniqueConstraintName.UNIQUE_FRAME_ID_PER_PROMPT,
            "frame_id",
            unique=True,
            sqlite_where=sa_text("frame_id IS NOT NULL"),
        ),
        # ensure text prompts have text, visual prompts have frame_id
        CheckConstraint(
            "(type = 'TEXT' AND text IS NOT NULL AND frame_id IS NULL) OR "
            "(type = 'VISUAL' AND frame_id IS NOT NULL AND text IS NULL)",
            name=CheckConstraintName.PROMPT_CONTENT,
        ),
    )


class ProcessorDB(Base):
    __tablename__ = "Processor"
    name: Mapped[str | None] = mapped_column(nullable=True)
    active: Mapped[bool] = mapped_column(nullable=False, default=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    project: Mapped["ProjectDB"] = relationship(back_populates="processors", single_parent=True)
    __table_args__ = (
        Index(
            UniqueConstraintName.PROCESSOR_NAME_PER_PROJECT,
            "project_id",
            "name",
            unique=True,
            sqlite_where=sa_text("name IS NOT NULL"),
        ),
        Index(
            UniqueConstraintName.SINGLE_ACTIVE_PROCESSOR_PER_PROJECT,
            "project_id",
            "active",
            unique=True,
            sqlite_where=sa_text("active IS 1"),
        ),
    )


class ProjectDB(Base):
    __tablename__ = "Project"
    name: Mapped[str] = mapped_column(nullable=False)
    active: Mapped[bool] = mapped_column(nullable=False, default=False)
    sources: Mapped[list[SourceDB]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    processors: Mapped[list[ProcessorDB]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    sinks: Mapped[list[SinkDB]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    prompts: Mapped[list[PromptDB]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    labels: Mapped[list[LabelDB]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    __table_args__ = (
        UniqueConstraint("name", name=UniqueConstraintName.PROJECT_NAME),
        Index(
            UniqueConstraintName.SINGLE_ACTIVE_PROJECT,
            "active",
            unique=True,
            sqlite_where=sa_text("active IS 1"),
        ),
    )
