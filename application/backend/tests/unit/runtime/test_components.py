#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.reader import SampleDatasetConfig, SourceType
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
        patch("runtime.components.ProjectService") as svc_cls,
        patch("runtime.components.get_settings", return_value=settings),
        patch("runtime.components.Processor") as processor_cls,
    ):
        svc_cls.return_value.get_pipeline_config.return_value = cfg

        factory.create_processor(project_id, reference_batch)

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


def test_create_source_passes_dataset_paths_to_stream_reader_factory():
    project_id = uuid4()
    reader_cfg = SampleDatasetConfig(source_type=SourceType.SAMPLE_DATASET, dataset_id=None)
    cfg = PipelineConfig(project_id=project_id, device="xpu", reader=reader_cfg, processor=None, writer=None)
    dataset_paths = {uuid4(): Path("/tmp/datasets/aquarium")}
    source_reader = Mock(name="stream_reader")

    factory = DefaultComponentFactory(session_factory=FakeSessionFactory(), dataset_paths=dataset_paths)

    with (
        patch("runtime.components.ProjectService") as svc_cls,
        patch("runtime.components.StreamReaderFactory.create", return_value=source_reader) as create_reader,
        patch("runtime.components.Source") as source_cls,
    ):
        svc_cls.return_value.get_pipeline_config.return_value = cfg

        factory.create_source(project_id)

        create_reader.assert_called_once_with(reader_cfg, dataset_paths=dataset_paths)
        source_cls.assert_called_once_with(source_reader)
