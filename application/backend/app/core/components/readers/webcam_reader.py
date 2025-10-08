#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from core.components.readers.video_stream_reader import BaseOpenCVReader
from core.components.schemas.reader import ReaderConfig


class WebCamReader(BaseOpenCVReader):
    def __init__(self, config: ReaderConfig) -> None:
        super().__init__(source=config.device_id, config=config)
