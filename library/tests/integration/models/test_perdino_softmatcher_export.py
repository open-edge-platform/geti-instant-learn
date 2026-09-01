# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for PerDino / SoftMatcher export + OpenVINO sibling inference.

Mirrors ``test_matcher_export.py`` but exercises the migrated PerDino and
SoftMatcher models end-to-end: fit references, export the baked IR
(``model.xml`` / ``model.bin`` / ``metadata.json``), then run inference through
the ``PerDinoOpenVINO`` / ``SoftMatcherOpenVINO`` loaders and assert the
``Prediction`` contract.
"""

from pathlib import Path

import numpy as np
import pytest

from instantlearn.data.base import Batch
from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Sample
from instantlearn.data.torch.folder import FolderDataset
from instantlearn.data.torch.image import read_image
from instantlearn.models.per_dino import PerDino, PerDinoOpenVINO
from instantlearn.models.soft_matcher import SoftMatcher, SoftMatcherOpenVINO
from instantlearn.utils.constants import SAMModelName
from tests import CPU_DEVICE


@pytest.fixture
def fss1000_root() -> Path:
    """Return path to the fss-1000 test dataset."""
    return Path(__file__).parent.parent.parent.parent / "examples" / "assets" / "fss-1000"


@pytest.fixture
def reference_batch(fss1000_root: Path) -> Batch:
    """Get a 1-shot reference batch from the apple category."""
    dataset = FolderDataset(root=fss1000_root, categories=["apple"], n_shots=1)
    return Batch.collate([dataset.get_reference_dataset()[0]])


@pytest.fixture
def target_sample(fss1000_root: Path) -> Sample:
    """Return a target Sample (numpy HWC image) from the apple category."""
    img = read_image(fss1000_root / "images" / "apple" / "2.jpg").numpy()  # CHW
    return Sample(image=np.ascontiguousarray(img.transpose(1, 2, 0)))


@pytest.mark.parametrize(
    ("model_cls", "ov_cls"),
    [
        (PerDino, PerDinoOpenVINO),
        (SoftMatcher, SoftMatcherOpenVINO),
    ],
)
def test_export_openvino_and_sibling_inference(
    model_cls: type,
    ov_cls: type,
    reference_batch: Batch,
    target_sample: Sample,
    tmp_path: Path,
) -> None:
    """Fit, export the baked IR, and run inference through the OV sibling."""
    pytest.importorskip("openvino")

    model = model_cls(
        sam=SAMModelName.SAM_HQ_TINY,
        device=CPU_DEVICE,
        precision="fp32",
        encoder_model="dinov3_small",
    )
    model.fit(reference_batch)

    export_dir = model.to_openvino(tmp_path)

    assert export_dir.is_dir()
    assert (export_dir / "model.xml").exists()
    assert (export_dir / "model.bin").exists()
    assert (export_dir / "metadata.json").exists()

    ov_model = ov_cls(model_dir=export_dir, device=CPU_DEVICE)
    predictions = ov_model.predict([target_sample])

    assert isinstance(predictions, list)
    assert len(predictions) == 1
    pred = predictions[0]
    assert isinstance(pred, Prediction)
    # Semantic-segmentation export yields one mask per baked category.
    assert pred.masks.shape[0] == pred.scores.shape[0] == pred.label_ids.shape[0]
    # Masks are resized back to the original frame.
    assert pred.masks.shape[1:] == target_sample.image.shape[:2]

    # References are baked into the IR: fit() is unsupported on the OV sibling.
    with pytest.raises(NotImplementedError):
        ov_model.fit(reference_batch)
