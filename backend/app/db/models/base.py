# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    __abstract__ = True
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
