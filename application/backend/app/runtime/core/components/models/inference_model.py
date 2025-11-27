from uuid import UUID
import numpy as np
import torch
from getiprompt.data.base.batch import Batch
from getiprompt.models.base import Model
import logging
from runtime.core.components.base import ModelHandler

logger=logging.getLogger(__name__)


class InferenceModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch) -> None:
        self._model = model
        self._reference_batch = reference_batch
        self._category_to_label_id = {
            idx: UUID(cat)
            for idx, cat in enumerate(reference_batch.categories[0])  # todo
        }

    def initialise(self) -> None:
        self._model.learn(self._reference_batch)

    def infer(self, batch: Batch) -> list[dict[str, torch.Tensor]]:

        logger.info("InferenceModelHandler infer called with batch of size %d", len(batch.samples))
        results = self._model.infer(batch)


        # results = []
        # for sample in batch:
        #     # Convert image to CHW tensor if needed
        #     if isinstance(sample.image, np.ndarray):
        #         image = torch.from_numpy(sample.image).permute(2, 0, 1)
        #     else:
        #         image = sample.image
        #
        #     # Run inference
        #     output = self._model(
        #         image=image.unsqueeze(0),
        #         masks=sample.masks.unsqueeze(0) if sample.masks is not None else None,
        #     )
        #
        #     # Convert label IDs (UUIDs as strings) to integer tensor
        #     if sample.categories:
        #         # Create mapping from category to index
        #         unique_categories = sorted(set(sample.categories))
        #         category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        #         label_indices = [category_to_idx[cat] for cat in sample.categories]
        #         pred_labels = torch.tensor(label_indices, dtype=torch.long)
        #     elif sample.category_ids is not None:
        #         # Use category_ids directly if available
        #         if isinstance(sample.category_ids, np.ndarray):
        #             pred_labels = torch.from_numpy(sample.category_ids).long()
        #         else:
        #             pred_labels = sample.category_ids.long()
        #     else:
        #         # No labels available, use zeros
        #         num_predictions = output.get("pred_masks", output.get("pred_boxes")).shape[0]
        #         pred_labels = torch.zeros(num_predictions, dtype=torch.long)
        #
        #     results.append({
        #         "pred_masks": output.get("pred_masks"),
        #         "pred_boxes": output.get("pred_boxes"),
        #         "pred_labels": pred_labels,
        #     })

        return results
