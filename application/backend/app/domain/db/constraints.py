# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum


class UniqueConstraintName(StrEnum):
    """Database unique constraint names."""

    PROJECT_NAME = "uq_project_name"
    SINGLE_TEXT_PROMPT_PER_PROJECT = "uq_single_text_prompt_per_project"
    PROCESSOR_NAME_PER_PROJECT = "uq_processor_name_per_project"
    SINGLE_ACTIVE_PROCESSOR_PER_PROJECT = "uq_single_active_processor_per_project"
    SOURCE_NAME_PER_PROJECT = "uq_source_name_per_project"
    SOURCE_TYPE_PER_PROJECT = "uq_source_type_per_project"
    LABEL_NAME_PER_PROJECT = "uq_label_name_per_project"
    SINGLE_ACTIVE_PROJECT = "uq_single_active_project"
    SINGLE_CONNECTED_SOURCE_PER_PROJECT = "uq_single_connected_source_per_project"
    UNIQUE_FRAME_ID_PER_PROMPT = "uq_unique_frame_id_per_prompt"
    SINK_NAME_PER_PROJECT = "uq_sink_name_per_project"
    SINK_TYPE_PER_PROJECT = "uq_sink_type_per_project"
    SINGLE_ACTIVE_SINK_PER_PROJECT = "uq_single_active_sink_per_project"


class CheckConstraintName(StrEnum):
    """Database check constraint names."""

    PROMPT_CONTENT = "ck_prompt_content"
    LABEL_PARENT = "ck_label_parent"
