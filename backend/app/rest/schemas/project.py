# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field

from rest.schemas.processor import ProcessorSchema
from rest.schemas.sink import SinkSchema
from rest.schemas.source import SourceSchema

class ProjectSchema(BaseModel):
    id: UUID
    name: str = Field(max_length=80, min_length=1)
    source: SourceSchema | None = None
    processor: ProcessorSchema | None = None
    sink: SinkSchema | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "4303eb19-dea8-4b67-be2d-054db0baed61",
                "name": "Cool project that does cool things",
                "source": {
                   "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                   "type": "VIDEO_FILE",
                   "config": {
                     "additionalProp1": {}
                   }
                },
                "processor": {
                  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                  "type": "DUMMY",
                  "config": {
                    "additionalProp1": {}
                  },
                  "name": "string"
                },
                "sink": {
                  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                  "config": {
                    "additionalProp1": {}
                  }
                }
            }
        }
    }

class ProjectPostPayload(BaseModel):
    id: UUID | None = None
    name: str = Field(max_length=80, min_length=1)


class ProjectPutPayload(BaseModel):
    name: str = Field(max_length=80, min_length=1)


class ProjectListItem(BaseModel):
    id: UUID
    name: str

class ProjectsListSchema(BaseModel):
    projects: list[ProjectListItem]

    model_config = {
        "json_schema_extra": {
            "example": {
                "projects": [
                    {
                        "id": "4303eb19-dea8-4b67-be2d-054db0baed61",
                        "name": "Cool project that does cool things",
                    }
                ]
            }
        }
    }


