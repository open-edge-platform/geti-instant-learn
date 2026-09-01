import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from datumaro import Dataset, DatasetItem, Image
from datumaro.components.annotation import Bbox, Mask, Points

from domain.services.schemas.processor import OutputData
from domain.services.schemas.writer import DatasetConfig
from runtime.core.components.base import StreamWriter

logger = logging.getLogger(__name__)

class DatasetWriter(StreamWriter):
    def __init__(self, config: DatasetConfig) -> None:
        self._config = config
        self._frame_count: int = 0
        self._buffered_frame_count: int = 0
        self._chunk_index: int = 0
        self._chunk_size: int | None = config.export_chunk_size

        # TODO: Consider making this configurable so multiple chunks can
        # export in parallel when disk I/O is slow.
        self._export_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dataset-export",
        )
        self._export_futures: list[Future[None]] = []

        if self._config.dataset_format is None:
            raise ValueError("dataset_format must be set")
        if self._config.output_dir is None:
            raise ValueError("output_dir must be set")
        self._dataset = Dataset()
        logger.info("DatasetWriter ready. Dataset format: %s, Output dir: %s", self._config.dataset_format, self._config.output_dir)

    def _get_export_dir(self) -> Path:
        output_dir = Path(self._config.output_dir)
        if self._chunk_size is None:
            return output_dir
        return output_dir / f"batch_{self._chunk_index:04d}"

    def _reset_buffer(self) -> None:
        self._dataset = Dataset()
        self._buffered_frame_count = 0

    def _flush_chunk_if_needed(self) -> None:
        if self._chunk_size is None:
            return
        if self._buffered_frame_count < self._chunk_size:
            return
        self._queue_export(
            dataset=self._dataset,
            output_dir=self._get_export_dir(),
            buffered_frame_count=self._buffered_frame_count,
            total_frame_count=self._frame_count,
        )
        self._chunk_index += 1
        self._reset_buffer()

    def _queue_export(
        self,
        dataset: Dataset,
        output_dir: Path,
        buffered_frame_count: int,
        total_frame_count: int,
    ) -> None:
        future = self._export_executor.submit(
            self._export,
            dataset,
            output_dir,
            buffered_frame_count,
            total_frame_count,
        )
        self._export_futures.append(future)

    def _drain_export_futures(self, wait: bool) -> None:
        pending_futures: list[Future[None]] = []
        for future in self._export_futures:
            if wait or future.done():
                future.result()
            else:
                pending_futures.append(future)
        self._export_futures = pending_futures

    def connect(self) -> None:
        # No connection needed for dataset writer
        pass

    def write(self, data: OutputData) -> None:
        """Buffer a single pipeline output into the Dataset.

        Processes OutputData.results which is list[dict[str, np.ndarray]]
        where each dict contains predictions for a single image.

        Handles Detection, Segmentation, and Point predictions based on the presence of keys:

            pred_masks  : Segmentation masks, shape [N, H, W].
            pred_points : Point predictions, shape [N, 4] as [x, y, score, fg_label].
            pred_boxes  : Bounding boxes, shape [N, 5] as [x1, y1, x2, y2, score].
            pred_labels : Class indices, shape [N].
            pred_scores : Confidence scores, shape [N].

        Args:
            data: OutputData containing frame and model predictions.

        Raises:
            ValueError: If OutputData.results is empty or not in expected format.
        """
        self._drain_export_futures(wait=False)

        if self._config.max_frames is not None and self._frame_count >= self._config.max_frames:
            return
        if not data.results:
            raise ValueError("OutputData.results is empty")

        for result in data.results:
            pred_labels = result.get("pred_labels")
            if pred_labels is None:
                raise ValueError("OutputData result is missing 'pred_labels'")
            annotations = []
            # Determine number of instances from pred_labels
            num_instances = len(pred_labels)
            # Process each instance
            for i in range(num_instances):
                label_id = int(pred_labels[i])
                # Add mask if available
                if "pred_masks" in result:
                    mask = result["pred_masks"][i]
                    annotations.append(
                        Mask(
                            image=mask,
                            label=label_id,
                            attributes={"score": float(result["pred_scores"][i])} if "pred_scores" in result else None,
                        )
                    )
                # Add bbox if available
                if "pred_boxes" in result:
                    x1, y1, x2, y2, score = result["pred_boxes"][i].tolist()
                    annotations.append(
                        Bbox(
                            x=x1, y=y1,
                            w=x2 - x1, h=y2 - y1,
                            label=label_id,
                            attributes={"score": score},
                        )
                    )
                # Add points if available
                if "pred_points" in result:
                    x, y, score, fg_label = result["pred_points"][i].tolist()
                    annotations.append(
                        Points(
                            points=[(x, y)],
                            label=label_id,
                            attributes={"score": score, "fg_label": fg_label},
                        )
                    )

            item_id = f"frame_{self._frame_count:06d}"
            attributes = {}
            if self._config.frame_trace and data.trace:
                attributes["trace_id"] = data.trace.frame_id
            if self._config.category_mapping is not None:
                attributes["category_mapping"] = self._config.category_mapping

            item = DatasetItem(
                id=item_id,
                image=Image.from_numpy(data.frame),
                annotations=annotations,
                attributes=attributes or None,
            )
            self._dataset.put(item)
            logger.debug(
                "Frame %d buffered. Objects: %d",
                self._frame_count,
                num_instances,
            )
            self._frame_count += 1
            self._buffered_frame_count += 1
            self._flush_chunk_if_needed()

    def _export(
        self,
        dataset: Dataset,
        output_dir: Path,
        buffered_frame_count: int,
        total_frame_count: int,
    ) -> None:
        """Serialize a Dataset snapshot to disk.

        Raises:
            ValueError: If dataset_format is not provided or not supported.
            RuntimeError: If no frames have been captured.
        """
        fmt = self._config.dataset_format

        if not fmt:
            raise ValueError("dataset_format is not provided")

        if buffered_frame_count == 0:
            raise RuntimeError("No frames captured. Nothing to export.")

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            dataset.export(str(output_dir), format=fmt, save_media=True)
        except Exception as e:
            logger.error("Error occurred while exporting dataset: %s", e)
            raise ValueError(f"Failed to export dataset with format '{fmt}': {e}") from e
        logger.info(
            "Dataset exported to %s in %s format. Buffered frames: %d. Total frames: %d",
            output_dir, fmt, buffered_frame_count, total_frame_count,
        )

    def close(self) -> None:
        """Export the remaining buffer if configured, then release in-memory state."""
        released_frames = self._buffered_frame_count
        try:
            if self._buffered_frame_count > 0 and self._config.dataset_format:
                self._queue_export(
                    dataset=self._dataset,
                    output_dir=self._get_export_dir(),
                    buffered_frame_count=self._buffered_frame_count,
                    total_frame_count=self._frame_count,
                )
                if self._chunk_size is not None:
                    self._chunk_index += 1
            elif self._buffered_frame_count > 0:
                logger.warning(
                    "Skipping export during close because dataset_format is not set. Buffered frames: %d",
                    self._buffered_frame_count,
                )
            self._drain_export_futures(wait=True)
        finally:
            self._export_executor.shutdown(wait=True)
            self._reset_buffer()
            self._frame_count = 0
            self._chunk_index = 0
            logger.info("DatasetWriter closed. Released buffered frames: %d", released_frames)


