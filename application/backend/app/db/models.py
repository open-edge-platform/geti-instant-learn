# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, ForeignKey, Index, Text, UniqueConstraint, text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from db.constraints import CheckConstraintName, UniqueConstraintName


class Base(DeclarativeBase):
    __abstract__ = True
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)


class LabelDB(Base):
    __tablename__ = "Label"
    name: Mapped[str] = mapped_column(nullable=False)
    color: Mapped[str] = mapped_column(nullable=False)
    project_id: Mapped[UUID | None] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    prompt_id: Mapped[UUID | None] = mapped_column(ForeignKey("Prompt.id", ondelete="CASCADE"))
    prompt: Mapped["PromptDB"] = relationship(back_populates="labels")
    project: Mapped["ProjectDB"] = relationship(back_populates="labels")
    __table_args__ = (
        CheckConstraint(
            "project_id IS NOT NULL OR prompt_id IS NOT NULL",
            name=CheckConstraintName.LABEL_PARENT,
        ),
        Index(
            UniqueConstraintName.LABEL_NAME_PER_PROJECT,
            "name",
            "project_id",
            unique=True,
        ),
    )


class AnnotationDB(Base):
    __tablename__ = "Annotation"
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    prompt_id: Mapped[UUID] = mapped_column(ForeignKey("Prompt.id", ondelete="CASCADE"))
    prompt: Mapped["PromptDB"] = relationship(back_populates="annotations", single_parent=True)
    __table_args__ = (UniqueConstraint("prompt_id"),)


class SourceDB(Base):
    __tablename__ = "Source"
    connected: Mapped[bool] = mapped_column(nullable=False, default=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    project: Mapped["ProjectDB"] = relationship(back_populates="sources")
    __table_args__ = (
        Index(
            UniqueConstraintName.SOURCE_TYPE_PER_PROJECT,
            "project_id",
            text("json_extract(config, '$.source_type')"),
            unique=True,
        ),
        Index(
            UniqueConstraintName.SOURCE_NAME_PER_PROJECT,
            "project_id",
            text("json_extract(config, '$.name')"),
            unique=True,
            sqlite_where=text("json_extract(config, '$.name') IS NOT NULL"),
        ),
        Index(
            UniqueConstraintName.SINGLE_CONNECTED_SOURCE_PER_PROJECT,
            "project_id",
            "connected",
            unique=True,
            sqlite_where=text("connected IS 1"),
        ),
    )


class SinkDB(Base):
    __tablename__ = "Sink"
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    project: Mapped["ProjectDB"] = relationship(back_populates="sinks", single_parent=True)


class PromptType(StrEnum):
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
    annotations: Mapped[list[AnnotationDB]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan", passive_deletes=True
    )
    labels: Mapped[list[LabelDB]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan", passive_deletes=True
    )
    __table_args__ = (
        UniqueConstraint(
            "name",
            "project_id",
            name=UniqueConstraintName.PROMPT_NAME_PER_PROJECT,
        ),
    )


class ProcessorDB(Base):
    __tablename__ = "Processor"
    name: Mapped[str | None] = mapped_column(nullable=True)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("Project.id", ondelete="CASCADE"))
    project: Mapped["ProjectDB"] = relationship(back_populates="processors", single_parent=True)
    __table_args__ = (
        Index(
            UniqueConstraintName.PROCESSOR_NAME_PER_PROJECT,
            "project_id",
            "name",
            unique=True,
            sqlite_where=text("name IS NOT NULL"),
        ),
    )


class ProjectDB(Base):
    __tablename__ = "Project"
    name: Mapped[str] = mapped_column(nullable=False)
    active: Mapped[bool] = mapped_column(nullable=False, default=False)
    sources: Mapped[list[SourceDB]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    processors: Mapped[list[ProcessorDB]] = relationship(back_populates="project")
    sinks: Mapped[list[SinkDB]] = relationship(back_populates="project")
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
            sqlite_where=text("active IS 1"),
        ),
    )
