# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LinearSumAssignment."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment as scipy_lsa

from instantlearn.components.linear_sum_assignment import LinearSumAssignment, linear_sum_assignment


class TestLinearSumAssignment:
    """Core tests for LinearSumAssignment."""

    @pytest.mark.parametrize("maximize", [True, False])
    def test_greedy_optimality(self, maximize: bool) -> None:
        """Greedy should produce valid assignments within acceptable range of scipy."""
        torch.manual_seed(42)
        for size in [(5, 5), (8, 12), (12, 8), (100, 50)]:
            cost = torch.rand(*size)
            row, col = LinearSumAssignment(maximize=maximize, method="greedy")(cost)
            our_cost = cost[row, col].sum().item()

            scipy_row, scipy_col = scipy_lsa(cost.numpy(), maximize=maximize)
            scipy_cost = cost[scipy_row, scipy_col].sum().item()

            # Greedy should be within reasonable range (75% for all cases)
            # For rectangular matrices with maximize=True, it achieves 99%+
            ratio = our_cost / scipy_cost if maximize else scipy_cost / our_cost
            assert ratio >= 0.75, f"Greedy ratio {ratio:.2f} < 0.75 for {size}"

    def test_edge_cases(self) -> None:
        """Test empty and single element matrices."""
        matcher = LinearSumAssignment(maximize=True, method="greedy")

        # Empty
        row, col = matcher(torch.empty(0, 0))
        assert row.numel() == 0 and col.numel() == 0

        # Single element
        row, col = matcher(torch.tensor([[5.0]]))
        assert row.item() == 0 and col.item() == 0

    def test_functional_api(self) -> None:
        """Test functional interface matches scipy."""
        cost = torch.tensor([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=torch.float32)
        row, col = linear_sum_assignment(cost, maximize=False)

        scipy_row, scipy_col = scipy_lsa(cost.numpy(), maximize=False)
        assert np.isclose(cost[row, col].sum().item(), cost[scipy_row, scipy_col].sum().item())


class TestExport:
    """Test export to TorchScript, ONNX, and OpenVINO."""

    @pytest.fixture
    def matcher(self) -> LinearSumAssignment:
        """Create greedy matcher fixture."""
        return LinearSumAssignment(maximize=True, method="greedy")

    def test_onnx_export_and_inference(self, matcher: LinearSumAssignment) -> None:
        """Test ONNX export and runtime inference."""
        onnx = pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")

        torch.manual_seed(42)
        cost = torch.rand(5, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "matcher.onnx"
            torch.onnx.export(
                matcher,
                cost,
                str(path),
                input_names=["cost"],
                output_names=["row", "col"],
                opset_version=17,
                dynamo=False,
            )

            # Validate ONNX model
            onnx.checker.check_model(onnx.load(str(path)))

            # Test inference
            sess = ort.InferenceSession(str(path))
            ort_row, ort_col = sess.run(None, {"cost": cost.numpy()})
            pt_row, pt_col = matcher(cost)

            assert np.array_equal(pt_row.numpy(), ort_row)
            assert np.array_equal(pt_col.numpy(), ort_col)

    def test_openvino_inference(self, matcher: LinearSumAssignment) -> None:
        """Test OpenVINO export and inference."""
        ov = pytest.importorskip("openvino")

        torch.manual_seed(42)
        cost = torch.rand(5, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "matcher.onnx"
            torch.onnx.export(
                matcher,
                cost,
                str(path),
                input_names=["cost"],
                output_names=["row", "col"],
                opset_version=17,
                dynamo=False,
            )

            compiled = ov.Core().compile_model(str(path), "CPU")
            ov_result = compiled({0: cost.numpy()})
            ov_row, ov_col = ov_result[0], ov_result[1]

            pt_row, pt_col = matcher(cost)
            assert np.array_equal(pt_row.numpy(), ov_row)
            assert np.array_equal(pt_col.numpy(), ov_col)


class TestDevicesAndDtypes:
    """Test device and dtype compatibility."""

    def test_cpu_and_dtypes(self) -> None:
        """Test CPU with various dtypes."""
        matcher = LinearSumAssignment(maximize=True, method="greedy")
        for dtype in [torch.float32, torch.float64]:
            cost = torch.rand(5, 5, dtype=dtype)
            row, _col = matcher(cost)
            assert row.device.type == "cpu" and row.numel() == 5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self) -> None:
        """Test CUDA device."""
        matcher = LinearSumAssignment(maximize=True, method="greedy")
        cost = torch.rand(5, 5, device="cuda")
        row, _col = matcher(cost)
        assert row.device.type == "cuda" and row.numel() == 5


class TestAutoMode:
    """Test auto mode behavior."""

    def test_auto_uses_scipy_normally(self) -> None:
        """Auto mode should match scipy during normal execution."""
        torch.manual_seed(42)
        cost = torch.rand(10, 10)

        solver = LinearSumAssignment(maximize=True, method="auto")
        row, col = solver(cost)

        scipy_row, scipy_col = scipy_lsa(cost.numpy(), maximize=True)
        assert np.isclose(cost[row, col].sum().item(), cost[scipy_row, scipy_col].sum().item())

    def test_auto_exports_to_onnx(self) -> None:
        """Auto mode should export successfully to ONNX."""
        pytest.importorskip("onnx")

        solver = LinearSumAssignment(maximize=True, method="auto")
        cost = torch.rand(5, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "auto.onnx"
            torch.onnx.export(solver, cost, str(path), opset_version=17, dynamo=False)
            assert path.exists()
