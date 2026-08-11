# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Matcher export methods with real model inference.

This module tests the full export flow including:
- Exporting to ONNX and OpenVINO formats
- Running inference with exported models
- Validating output shapes and values
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from instantlearn.data.base import Batch
from instantlearn.data.torch.folder import FolderDataset
from instantlearn.data.torch.image import read_image
from instantlearn.models.matcher import Matcher
from instantlearn.utils.constants import SAMModelName
from tests import CPU_DEVICE


@pytest.fixture
def fss1000_root() -> Path:
    """Return path to fss-1000 test dataset."""
    return Path(__file__).parent.parent.parent.parent / "examples" / "assets" / "fss-1000"


@pytest.fixture
def reference_image_path(fss1000_root: Path) -> Path:
    """Return path to reference image."""
    return fss1000_root / "images" / "apple" / "1.jpg"


@pytest.fixture
def target_image_path(fss1000_root: Path) -> Path:
    """Return path to target image."""
    return fss1000_root / "images" / "apple" / "2.jpg"


@pytest.fixture
def reference_mask_path(fss1000_root: Path) -> Path:
    """Return path to reference mask."""
    return fss1000_root / "masks" / "apple" / "1.png"


@pytest.fixture
def dataset(fss1000_root: Path) -> FolderDataset:
    """Create a FolderDataset for testing."""
    return FolderDataset(
        root=fss1000_root,
        categories=["apple"],
        n_shots=1,
    )


@pytest.fixture
def reference_batch(dataset: FolderDataset) -> Batch:
    """Get reference batch from dataset."""
    ref_dataset = dataset.get_reference_dataset()
    samples = [ref_dataset[0]]
    return Batch.collate(samples)


class TestMatcherExportIntegration:
    """Integration tests for Matcher export functionality."""

    @pytest.mark.parametrize("sam_model", [SAMModelName.SAM_HQ_BASE])
    def test_export_onnx_and_inference(
        self,
        sam_model: SAMModelName,
        reference_batch: Batch,
        target_image_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test exporting to ONNX and running inference.

        Args:
            sam_model: SAM model to use.
            reference_batch: Reference batch for fitting.
            target_image_path: Path to target image.
            tmp_path: Temporary directory for export.
        """
        # Initialize Matcher
        matcher = Matcher(
            sam=sam_model,
            device=CPU_DEVICE,
            precision="fp32",
            encoder_model="dinov3_small",
            use_mask_refinement=False,
        )

        # Fit on reference
        matcher.fit(reference_batch)

        # Export to ONNX
        exported_path = matcher.to_onnx(tmp_path)

        # Verify file exists
        assert exported_path.exists()
        assert exported_path.suffix == ".onnx"

        # Run inference with ONNX Runtime
        target_image = read_image(target_image_path)
        session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])

        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        # Prepare input: add batch dimension and convert to numpy
        input_data = target_image.numpy()[None, ...].astype(np.float32)

        # The ONNX model is exported with a static spatial size (the encoder's
        # input_size, e.g. 512x512). Resize the input if it doesn't match.
        expected_shape = session.get_inputs()[0].shape  # [1, 3, H, W]
        expected_h, expected_w = int(expected_shape[2]), int(expected_shape[3])
        if input_data.shape[2] != expected_h or input_data.shape[3] != expected_w:
            tensor = torch.from_numpy(input_data)
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(expected_h, expected_w),
                mode="bilinear",
                align_corners=False,
            )
            input_data = tensor.numpy()

        outputs = session.run(output_names, {input_name: input_data})

        masks, scores, labels = outputs

        # Validate output shapes
        assert masks.ndim == 3, f"Expected masks to have 3 dims, got {masks.ndim}"
        assert scores.ndim == 1, f"Expected scores to have 1 dim, got {scores.ndim}"
        assert labels.ndim == 1, f"Expected labels to have 1 dim, got {labels.ndim}"
        assert masks.shape[0] == scores.shape[0] == labels.shape[0], "Output counts should match"

        # Validate mask spatial dimensions match model's static input size
        assert masks.shape[1] == expected_h, "Mask height should match model input"
        assert masks.shape[2] == expected_w, "Mask width should match model input"

    @pytest.mark.parametrize("sam_model", [SAMModelName.SAM_HQ_BASE])
    def test_export_openvino_and_inference(
        self,
        sam_model: SAMModelName,
        reference_batch: Batch,
        target_image_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test exporting to OpenVINO and running inference.

        Args:
            sam_model: SAM model to use.
            reference_batch: Reference batch for fitting.
            target_image_path: Path to target image.
            tmp_path: Temporary directory for export.
        """
        openvino = pytest.importorskip("openvino")

        # Initialize Matcher
        matcher = Matcher(
            sam=sam_model,
            device=CPU_DEVICE,
            precision="fp32",
            encoder_model="dinov3_small",
            use_mask_refinement=False,
        )

        # Fit on reference
        matcher.fit(reference_batch)

        # Export to OpenVINO (returns the IR directory)
        exported_dir = matcher.to_openvino(tmp_path)

        # Verify files exist
        assert exported_dir.is_dir()
        ir_path = exported_dir / "model.xml"
        assert ir_path.exists()
        assert (exported_dir / "model.bin").exists()

        # Run inference with OpenVINO
        target_image = read_image(target_image_path)
        core = openvino.Core()
        ov_model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(ov_model, "CPU")

        # Resize input to match the model's static input shape.
        expected_shape = compiled_model.input(0).shape
        input_data = target_image.numpy()[None, ...].astype(np.float32)
        if input_data.shape != tuple(expected_shape):
            tensor = torch.from_numpy(input_data)
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(expected_shape[2], expected_shape[3]),
                mode="bilinear",
            )
            input_data = tensor.numpy()

        outputs = compiled_model(input_data)

        masks, scores, labels = outputs.values()

        # Validate output shapes
        assert masks.ndim == 3, f"Expected masks to have 3 dims, got {masks.ndim}"
        assert scores.ndim == 1, f"Expected scores to have 1 dim, got {scores.ndim}"
        assert labels.ndim == 1, f"Expected labels to have 1 dim, got {labels.ndim}"
        assert masks.shape[0] == scores.shape[0] == labels.shape[0], "Output counts should match"

        # Validate mask spatial dimensions match model input (static shape export)
        assert masks.shape[1] == expected_shape[2], "Mask height should match model input"
        assert masks.shape[2] == expected_shape[3], "Mask width should match model input"
