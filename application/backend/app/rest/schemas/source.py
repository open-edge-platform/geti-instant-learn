# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from core.components.schemas.reader import ReaderConfig, SourceType


class WebcamSourcePayload(BaseModel):
    source_type: Literal[SourceType.WEBCAM]  # type: ignore[valid-type]
    name: str = "Default Webcam"
    device_id: int


InputSourcePayload = Union[WebcamSourcePayload]  # noqa UP007, extend this union later
SourcePayloadSchema = Annotated[InputSourcePayload, Field(discriminator="source_type")]


class SourceSchema(ReaderConfig):
    connected: bool


class SourcesListSchema(BaseModel):
    sources: list[SourceSchema]
