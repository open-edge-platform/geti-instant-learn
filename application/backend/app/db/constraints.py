# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum


class UniqueConstraintName(StrEnum):
    """Database unique constraint names."""

    PROJECT_NAME = "uq_project_name"
    PROMPT_NAME_PER_PROJECT = "uq_prompt_name_per_project"
    PROCESSOR_NAME_PER_PROJECT = "uq_processor_name_per_project"
    SOURCE_NAME_PER_PROJECT = "uq_source_name_per_project"
    SOURCE_TYPE_PER_PROJECT = "uq_source_type_per_project"
    LABEL_NAME_PER_PROJECT = "uq_label_name_per_project"
    SINGLE_ACTIVE_PROJECT = "uq_single_active_project"
    SINGLE_CONNECTED_SOURCE_PER_PROJECT = "uq_single_connected_source_per_project"


class CheckConstraintName(StrEnum):
    """Database check constraint names."""

    LABEL_PARENT = "ck_label_parent"
