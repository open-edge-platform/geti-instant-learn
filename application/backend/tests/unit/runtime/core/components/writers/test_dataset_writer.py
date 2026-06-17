import numpy as np
import pytest
from datumaro import Dataset

from domain.services.schemas.frame_trace import FrameTrace
from domain.services.schemas.processor import OutputData
from domain.services.schemas.writer import DatasetConfig
from runtime.core.components.writers.dataset_writer import DatasetWriter


@pytest.fixture
def export_calls(monkeypatch):
    """Replace datumaro's disk export with a recorder.

    Returns a list that receives one entry per export. Each entry captures the
    target directory and the DatasetItems that would have been serialized, so a
    test can assert *how many* exports happened, *where*, and *with how many
    frames* — without touching the filesystem or datumaro's format internals.
    """
    calls: list[dict] = []

    def _record(self, save_dir, format=None, save_media=False, **kwargs):  # noqa: A002 - mirrors datumaro's signature
        calls.append(
            {
                "save_dir": str(save_dir),
                "format": format,
                "items": list(self),
            }
        )

    monkeypatch.setattr(Dataset, "export", _record)
    return calls


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
    def test_buffers_in_memory_and_exports_once_on_close(self, config, export_calls):
        # No export_chunk_size -> everything is held in memory until close().
        writer = DatasetWriter(config=config)

        for _ in range(3):
            writer.write(make_output())

        # Nothing is written to disk while buffering.
        assert export_calls == []
        assert writer._frame_count == 3

        writer.close()

        # close() flushes the whole buffer in a single export to output_dir.
        assert len(export_calls) == 1
        assert export_calls[0]["save_dir"] == str(config.output_dir)
        assert len(export_calls[0]["items"]) == 3

    def test_chunking_writes_sequential_batch_dirs(self, config, export_calls, tmp_path):
        # With a chunk size of 2, every 2 frames are flushed to their own batch dir.
        config.export_chunk_size = 2
        writer = DatasetWriter(config=config)

        for _ in range(5):
            writer.write(make_output())
        writer.close()

        # 5 frames at chunk size 2 -> batches of 2, 2, and a final 1 from close().
        save_dirs = [call["save_dir"] for call in export_calls]
        item_counts = [len(call["items"]) for call in export_calls]
        assert save_dirs == [
            str(tmp_path / "batch_0000"),
            str(tmp_path / "batch_0001"),
            str(tmp_path / "batch_0002"),
        ]
        assert item_counts == [2, 2, 1]

    def test_max_frames_caps_buffered_frames(self, config, export_calls):
        # max_frames is a hard ceiling: writes past the limit are ignored.
        config.max_frames = 2
        writer = DatasetWriter(config=config)

        for _ in range(5):
            writer.write(make_output())
        writer.close()

        assert writer._frame_count == 0  # reset on close
        assert sum(len(call["items"]) for call in export_calls) == 2

    def test_frame_trace_id_is_recorded_as_attribute(self, config, export_calls):
        config.frame_trace = True
        trace = FrameTrace.create()
        writer = DatasetWriter(config=config)

        writer.write(make_output(trace=trace))
        writer.close()

        item = export_calls[0]["items"][0]
        assert item.attributes["trace_id"] == trace.frame_id

    def test_write_raises_on_empty_results(self, config):
        writer = DatasetWriter(config=config)

        with pytest.raises(ValueError, match="results is empty"):
            writer.write(OutputData(frame=np.zeros((2, 2, 3), dtype=np.uint8), results=[]))

    def test_write_raises_when_pred_labels_missing(self, config):
        writer = DatasetWriter(config=config)
        bad = OutputData(frame=np.zeros((2, 2, 3), dtype=np.uint8), results=[{"pred_scores": np.array([0.9])}])

        with pytest.raises(ValueError, match="missing 'pred_labels'"):
            writer.write(bad)

    def test_close_without_frames_does_not_export(self, config, export_calls):
        writer = DatasetWriter(config=config)

        writer.close()

        assert export_calls == []

    def test_usable_as_context_manager(self, config, export_calls):
        # StreamWriter is a context manager; __exit__ calls close().
        with DatasetWriter(config=config) as writer:
            writer.write(make_output())

        assert len(export_calls) == 1
        assert len(export_calls[0]["items"]) == 1
