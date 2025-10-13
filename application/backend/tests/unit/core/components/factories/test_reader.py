#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
import uuid

from core.components.factories.reader import StreamReaderFactory
from core.components.readers.noop_reader import NoOpReader
from core.components.readers.webcam_reader import WebCamReader
from core.components.schemas.reader import SourceType, WebCamConfig


class TestReader:
    def test_factory_returns_webcam_reader(self):
        webcam_config = WebCamConfig(source_type=SourceType.WEBCAM, id=uuid.uuid4(), device_id=1, name="web-cam")

        result = StreamReaderFactory.create(webcam_config)

        assert isinstance(result, WebCamReader)
        assert result.source_type == webcam_config.source_type
        assert result.source == webcam_config.device_id
        assert result.config == webcam_config

    def test_factory_returns_noop_reader_for_other_config(self):
        result = StreamReaderFactory.create(None)

        assert isinstance(result, NoOpReader)
