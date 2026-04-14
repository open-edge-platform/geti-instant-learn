# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from domain.services.schemas.mappers.source import source_schema_to_db
from domain.services.schemas.reader import SampleDatasetConfig, SourceType
from domain.services.schemas.source import SourceCreateSchema


def test_source_schema_to_db_serializes_sample_dataset_uuid_to_json_string() -> None:
    source_id = uuid4()
    project_id = uuid4()
    dataset_id = uuid4()

    schema = SourceCreateSchema(
        id=source_id,
        active=True,
        config=SampleDatasetConfig(
            source_type=SourceType.SAMPLE_DATASET,
            seekable=True,
            dataset_id=dataset_id,
        ),
    )

    db_source = source_schema_to_db(schema=schema, project_id=project_id)

    assert db_source.id == source_id
    assert db_source.project_id == project_id
    assert db_source.config["source_type"] == "sample_dataset"
    assert db_source.config["dataset_id"] == str(dataset_id)
