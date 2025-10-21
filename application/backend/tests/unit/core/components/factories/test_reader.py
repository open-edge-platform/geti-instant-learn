#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from core.components.factories.reader import StreamReaderFactory
from core.components.readers.noop_reader import NoOpReader
from core.components.readers.webcam_reader import WebCamReader
from core.components.schemas.reader import SourceType, WebCamConfig


class TestReader:
    def test_factory_returns_webcam_reader(self):
        webcam_config = WebCamConfig(source_type=SourceType.WEBCAM, device_id=1)

        result = StreamReaderFactory.create(webcam_config)

        assert isinstance(result, WebCamReader)
        assert result._config == webcam_config

    def test_factory_returns_noop_reader_for_other_config(self):
        result = StreamReaderFactory.create(None)

        assert isinstance(result, NoOpReader)
