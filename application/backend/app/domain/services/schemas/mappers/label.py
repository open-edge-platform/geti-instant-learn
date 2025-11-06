# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import secrets
from collections.abc import Iterable

from domain.db.models import LabelDB
from domain.services.schemas.base import Pagination
from domain.services.schemas.label import LabelCreateSchema, LabelSchema, LabelsListSchema


def label_db_to_schema(label: LabelDB) -> LabelSchema:
    """
    Map a LabelDB ORM instance to a LabelSchema.
    """
    return LabelSchema(id=label.id, name=label.name, color=label.color)


def labels_db_to_list_items(
    labels: Iterable[LabelDB], total: int, offset: int = 0, limit: int = 20
) -> LabelsListSchema:
    """
    Map an iterable of LabelDB entities to LabelsListSchema with pagination metadata.

    Parameters:
        labels: Iterable of LabelDB entities to map
        total: Total number of labels available
        offset: Starting index of the returned items
        limit: Maximum number of items requested

    Returns:
        LabelsListSchema with mapped labels and pagination metadata
    """
    items = [label_db_to_schema(label) for label in labels]

    pagination = Pagination(
        count=len(items),
        total=total,
        offset=offset,
        limit=limit,
    )

    return LabelsListSchema(labels=items, pagination=pagination)


def label_schema_to_db(payload: LabelCreateSchema) -> LabelDB:
    """
    Create a new (unpersisted) LabelDB entity from a LabelCreateSchema.
    The caller (service layer) is responsible for adding it to the session,
    flushing, activation handling, and committing.
    """
    if payload.color is None:
        color_hex = random_color()
    else:
        color_hex = payload.color.as_hex("long")

    return LabelDB(id=payload.id, name=payload.name, color=color_hex)


def random_color() -> str:
    """
    Generate random color.
    """
    red, green, blue = (
        secrets.randbelow(256),
        secrets.randbelow(256),
        secrets.randbelow(256),
    )
    return f"#{red:02x}{green:02x}{blue:02x}"
