# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from runtime.core.components.schemas.reader import ReaderConfig
from runtime.core.components.base import StreamReader


class StreamReaderFactory:

    @classmethod
    def create(cls, config: ReaderConfig) -> StreamReader:
        pass
