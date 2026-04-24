#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.reader import SampleDatasetConfig, SourceType
from runtime.components import DefaultComponentFactory


def test_create_processor_passes_pipeline_device_to_model_factory():
    project_id = uuid4()
    cfg = PipelineConfig(project_id=project_id, device="xpu", reader=None, processor=None, writer=None)
    reference_batch = Mock(name="reference_batch")

    factory = DefaultComponentFactory()
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


def test_create_source_passes_dataset_resolver_to_stream_reader_factory():
    reader_cfg = SampleDatasetConfig(source_type=SourceType.SAMPLE_DATASET, dataset_id=None)
    dataset_resolver = Mock(name="dataset_resolver")
    source_reader = Mock(name="stream_reader")

    factory = DefaultComponentFactory(dataset_resolver=dataset_resolver)

    with (
        patch.object(factory._reader_factory, "create", return_value=source_reader) as create_reader,
        patch("runtime.components.Source") as source_cls,
    ):
        factory.create_source(reader_cfg)

        create_reader.assert_called_once_with(reader_cfg)
        source_cls.assert_called_once_with(source_reader)
