# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from db.model.label import Label
    from db.model.processor import Processor
    from db.model.prompt import Prompt
    from db.model.sink import Sink
    from db.model.source import Source


class Pipeline(Base):
    __tablename__ = "Pipeline"
    name: Mapped[str] = mapped_column(nullable=False)
    active: Mapped[bool] = mapped_column(nullable=False, default=False)
    source: Mapped["Source"] = relationship(back_populates="pipeline")
    processor: Mapped["Processor"] = relationship(back_populates="pipeline")
    sink: Mapped["Sink"] = relationship(back_populates="pipeline")

    prompts: Mapped[list["Prompt"]] = relationship(
        back_populates="pipeline", cascade="all, delete-orphan", passive_deletes=True
    )
    labels: Mapped[list["Label"]] = relationship(
        back_populates="pipeline", cascade="all, delete-orphan", passive_deletes=True
    )
