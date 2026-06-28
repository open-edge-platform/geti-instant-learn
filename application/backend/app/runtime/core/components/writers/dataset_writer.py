import logging
from pathlib import Path

from datumaro import Dataset, DatasetItem, Image
from datumaro.components.annotation import AnnotationType, LabelCategories, Bbox,  Mask, Points

from domain.services.schemas.processor import ErrorData, OutputData
from domain.services.schemas.writer import DatasetConfig
from runtime.core.components.base import StreamWriter

logger = logging.getLogger(__name__)

# TODO: Remove broad exception silencing once the sink layer is ready to handle non-recoverable errors.
class DatasetWriter(StreamWriter):
    def __init__(self, config: DatasetConfig, export_chunk_size: int | None = None) -> None:
        self._config = config
        self._frame_count: int = 0
        self._buffered_frame_count: int = 0
        self._chunk_index: int = 0
        self._chunk_size: int | None = export_chunk_size

        if self._config.dataset_format is None:
            raise ValueError("dataset_format must be set")
        if self._config.output_dir is None:
            raise ValueError("output_dir must be set")
        self._dataset = Dataset(media_type=Image)
        mapping = config.category_id_to_name
        self._categories = (
            {AnnotationType.label: LabelCategories.from_iterable(
                mapping[i] for i in range(max(mapping) + 1)
            )}
            if mapping else None
        )
        self._dataset = Dataset(media_type=Image, categories=self._categories)
        logger.info("DatasetWriter ready. Dataset format: %s, Output dir: %s", self._config.dataset_format, self._config.output_dir)

    def _get_export_dir(self) -> Path:
        output_dir = Path(self._config.output_dir)
        if self._chunk_size is None:
            return output_dir
        return output_dir / f"batch_{self._chunk_index:04d}"

    def _reset_buffer(self) -> None:
        self._dataset = Dataset(media_type=Image, categories=self._categories)
        self._buffered_frame_count = 0

    def _flush_chunk_if_needed(self) -> None:
        if self._chunk_size is None:
            return
        if self._buffered_frame_count < self._chunk_size:
            return
        self._export(
            dataset=self._dataset,
            output_dir=self._get_export_dir(),
            buffered_frame_count=self._buffered_frame_count,
            total_frame_count=self._frame_count,
        )
        self._chunk_index += 1
        self._reset_buffer()

    def connect(self) -> None:
        # No connection needed for dataset writer
        pass

    def write(self, data: OutputData | ErrorData) -> None:
        """Buffer a single pipeline output into the Dataset.

        Processes OutputData.results which is list[dict[str, np.ndarray]]
        where each dict contains predictions for a single image.

            pred_masks  : Segmentation masks, shape [N, H, W].
            pred_points : Point predictions, shape [N, 4] as [x, y, score, fg_label].
            pred_boxes  : Bounding boxes, shape [N, 5] as [x1, y1, x2, y2, score].
            pred_labels : Class indices, shape [N].
            pred_scores : Confidence scores, shape [N].

        Args:
            data: OutputData containing frame and model predictions, or an upstream ErrorData.
        """
        if isinstance(data, ErrorData):
            logger.warning("Received upstream error: %s", data.message)
            return

        if self._config.max_frames is not None and self._frame_count >= self._config.max_frames:
            return

        for result in data.results:
            try:
                pred_labels = result["pred_labels"]
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

                item = DatasetItem(
                    id=item_id,
                    media=Image.from_numpy(data.frame),
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
            except Exception:
                logger.exception("Failed to write frame %d to dataset. Skipping.", self._frame_count)

    def _export(
        self,
        dataset: Dataset,
        output_dir: Path,
        buffered_frame_count: int,
        total_frame_count: int,
    ) -> None:
        """Serialize a Dataset snapshot to disk."""
        fmt = self._config.dataset_format
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset.export(str(output_dir), format=fmt, save_media=True)
            logger.info(
                "Dataset exported to %s in %s format. Buffered frames: %d. Total frames: %d",
                output_dir, fmt, buffered_frame_count, total_frame_count,
            )
        except Exception:
            logger.exception("Failed to export dataset to %s.", output_dir)

    def close(self) -> None:
        """Export the remaining buffer if configured, then release in-memory state."""
        released_frames = self._buffered_frame_count
        try:
            if self._buffered_frame_count > 0 and self._config.dataset_format:
                self._export(
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
        finally:
            self._reset_buffer()
            self._frame_count = 0
            self._chunk_index = 0
            logger.info("DatasetWriter closed. Released buffered frames: %d", released_frames)


