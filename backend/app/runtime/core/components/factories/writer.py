# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from runtime.core.components.base import StreamWriter
from runtime.core.components.schemas.writer import WriterConfig


class StreamWriterFactory:

    @classmethod
    def create(cls, config: WriterConfig) -> StreamWriter:
        pass
