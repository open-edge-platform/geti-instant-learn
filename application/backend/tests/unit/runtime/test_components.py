#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.services.schemas.pipeline import PipelineConfig
from runtime.components import DefaultComponentFactory


class FakeSessionCtx:
    def __init__(self):
        self.session = Mock()

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSessionFactory:
    def __call__(self):
        return FakeSessionCtx()


def test_create_processor_passes_pipeline_device_to_model_factory():
    project_id = uuid4()
    cfg = PipelineConfig(project_id=project_id, device="xpu", reader=None, processor=None, writer=None)
    reference_batch = Mock(name="reference_batch")

    factory = DefaultComponentFactory(session_factory=FakeSessionFactory())
    factory._model_factory = Mock(name="model_factory")
    model_handler = Mock(name="model_handler")
    factory._model_factory.create.return_value = model_handler

    settings = SimpleNamespace(
        processor_batch_size=8,
        processor_frame_skip_interval=2,
        processor_frame_skip_amount=1,
    )

    with (
        patch("runtime.components.get_settings", return_value=settings),
        patch("runtime.components.Processor") as processor_cls,
    ):
        factory.create_processor(cfg, reference_batch)

        factory._model_factory.create.assert_called_once_with(
            reference_batch=reference_batch,
            config=None,
            configured_device="xpu",
        )
        processor_cls.assert_called_once_with(
            model_handler=model_handler,
            batch_size=8,
            frame_skip_interval=2,
            frame_skip_amount=1,
        )
