# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import secrets
from collections.abc import Iterable

from domain.db.models import LabelDB
from domain.services.schemas.base import Pagination
from domain.services.schemas.label import LabelCreateSchema, LabelSchema, LabelsListSchema, RGBColor, VisualizationLabel


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


def _hex_to_rgb_tuple(hex_color: str) -> RGBColor:
    """
    Convert RGB hex color to RGB tuple.
    """
    hex_value = hex_color.lstrip("#")
    r = int(hex_value[0:2], 16)
    g = int(hex_value[2:4], 16)
    b = int(hex_value[4:6], 16)
    return RGBColor(r, g, b)


def label_db_to_visualization_label(label: LabelDB) -> VisualizationLabel:
    """
    Map a LabelDB entity to VisualizationLabel.
    """
    return VisualizationLabel(id=label.id, color=_hex_to_rgb_tuple(label.color), object_name=label.name)
