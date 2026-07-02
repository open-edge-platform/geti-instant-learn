import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from domain.dispatcher import ComponentType
from domain.services.schemas.frame_trace import FrameTrace
from domain.services.schemas.processor import ErrorData, OutputData
from domain.services.schemas.writer import DatasetConfig
from runtime.core.components.writers.dataset_writer import DatasetWriter


@pytest.fixture
def mock_exporter():
    """A fake exporter that records calls without touching datumaro."""
    exporter = MagicMock()
    return exporter


@pytest.fixture
def config(tmp_path):
    """A minimal valid DatasetConfig writing to a pytest tmp directory."""
    return DatasetConfig(output_dir=str(tmp_path), dataset_format="coco")


def make_output(label: int = 0, score: float = 0.9, trace: FrameTrace | None = None) -> OutputData:
    """Build a one-frame OutputData carrying a single bounding-box prediction."""
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = {
        "pred_labels": np.array([label]),
        "pred_boxes": np.array([[0.0, 0.0, 1.0, 1.0, score]]),
        "pred_scores": np.array([score]),
    }
    return OutputData(frame=frame, results=[result], trace=trace)


class TestDatasetWriter:
    def test_buffers_in_memory_and_exports_once_on_close(self, config, mock_exporter):
        # No export_chunk_size -> everything is held in memory until close().
        writer = DatasetWriter(config=config, exporter=mock_exporter)

        for _ in range(3):
            writer.write(make_output())

        # Nothing is written to disk while buffering.
        mock_exporter.export.assert_not_called()
        assert writer._frame_count == 3

        writer.close()

        # close() flushes the whole buffer in a single export to output_dir.
        mock_exporter.export.assert_called_once()
        export_calls = mock_exporter.export.call_args_list
        assert export_calls[0].kwargs["fmt"] == config.dataset_format
        assert export_calls[0].kwargs["output_dir"] == Path(config.output_dir)
        assert len(export_calls[0].kwargs["dataset"].get_subset("default")) == 3


    def test_chunking_writes_sequential_batch_dirs(self, config, mock_exporter, tmp_path):
        # With a chunk size of 2, every 2 frames are flushed to their own batch dir.
        writer = DatasetWriter(config=config, exporter=mock_exporter, export_chunk_size=2)

        for _ in range(5):
            writer.write(make_output())
        writer.close()

        # 5 frames at chunk size 2 -> batches of 2, 2, and a final 1 from close().
        save_dirs = [call.kwargs["output_dir"] for call in mock_exporter.export.call_args_list]
        item_counts = [len(call.kwargs["dataset"].get_subset("default")) for call in mock_exporter.export.call_args_list]
        assert save_dirs == [
            Path(tmp_path) / "batch_0000",
            Path(tmp_path) / "batch_0001",
            Path(tmp_path) / "batch_0002",
        ]
        assert item_counts == [2, 2, 1]

    def test_max_frames_caps_buffered_frames(self, config, mock_exporter):
        # max_frames is a hard ceiling: writes past the limit are ignored.
        config.max_frames = 2
        writer = DatasetWriter(config=config, exporter=mock_exporter)

        for _ in range(5):
            writer.write(make_output())
        writer.close()

        assert writer._frame_count == 0  # reset on close
        assert sum(len(call.kwargs["dataset"].get_subset("default")) for call in mock_exporter.export.call_args_list) == 2

    def test_frame_trace_id_is_recorded_as_attribute(self, config, mock_exporter):
        config.frame_trace = True
        trace = FrameTrace.create()
        writer = DatasetWriter(config=config, exporter=mock_exporter)

        writer.write(make_output(trace=trace))
        writer.close()

        item = list(mock_exporter.export.call_args_list[0].kwargs["dataset"].get_subset("default"))[0]
        assert item.attributes["trace_id"] == trace.frame_id

    def test_write_ignores_empty_results(self, config, mock_exporter):
        # An empty results list is a valid "nothing to write" frame, not an error.
        writer = DatasetWriter(config=config, exporter=mock_exporter)

        writer.write(OutputData(frame=np.zeros((2, 2, 3), dtype=np.uint8), results=[]))
        writer.close()

        assert writer._frame_count == 0
        assert mock_exporter.export.call_args_list == []

    def test_write_ignores_error_data(self, config, mock_exporter):
        # Upstream errors flow through the same queue; the writer logs and skips them.
        writer = DatasetWriter(config=config, exporter=mock_exporter)

        writer.write(ErrorData(message="upstream failed", component=ComponentType.PROCESSOR))
        writer.close()

        assert writer._frame_count == 0
        assert mock_exporter.export.call_args_list == []

    def test_close_without_frames_does_not_export(self, config, mock_exporter):
        writer = DatasetWriter(config=config, exporter=mock_exporter)

        writer.close()

        assert mock_exporter.export.call_args_list == []

    def test_usable_as_context_manager(self, config, mock_exporter):
        # StreamWriter is a context manager; __exit__ calls close().
        with DatasetWriter(config=config, exporter=mock_exporter) as writer:
            writer.write(make_output())

        assert len(mock_exporter.export.call_args_list) == 1
        assert len(mock_exporter.export.call_args_list[0].kwargs["dataset"].get_subset("default")) == 1
