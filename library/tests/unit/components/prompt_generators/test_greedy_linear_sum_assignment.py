# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GreedyLinearSumAssignment.

This module tests the pure PyTorch implementation of linear sum assignment
against scipy's implementation for numerical correctness, and validates
export compatibility with ONNX and OpenVINO.
"""

import tempfile
import time
from logging import getLogger
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment

from getiprompt.components.prompt_generators import GreedyLinearSumAssignment

logger = getLogger(__name__)


class TestGreedyLinearSumAssignmentNumerical:
    """Numerical tests comparing greedy implementation with scipy."""

    @pytest.fixture
    def greedy_matcher_maximize(self) -> GreedyLinearSumAssignment:
        """Create a greedy matcher for maximization."""
        return GreedyLinearSumAssignment(maximize=True)

    @pytest.fixture
    def greedy_matcher_minimize(self) -> GreedyLinearSumAssignment:
        """Create a greedy matcher for minimization."""
        return GreedyLinearSumAssignment(maximize=False)

    def test_simple_diagonal_matrix_maximize(
        self,
        greedy_matcher_maximize: GreedyLinearSumAssignment,
    ) -> None:
        """Test with a simple diagonal matrix where optimal assignment is obvious."""
        # Diagonal matrix - optimal assignment should be diagonal
        cost_matrix = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        row_ind, col_ind = greedy_matcher_maximize(cost_matrix)

        # Should assign each row to its corresponding column
        assert torch.equal(row_ind, torch.tensor([0, 1, 2]))
        assert torch.equal(col_ind, torch.tensor([0, 1, 2]))

    def test_simple_diagonal_matrix_minimize(
        self,
        greedy_matcher_minimize: GreedyLinearSumAssignment,
    ) -> None:
        """Test minimization with inverse diagonal matrix."""
        # Inverse diagonal - optimal for minimization is the diagonal
        cost_matrix = torch.tensor([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])

        row_ind, col_ind = greedy_matcher_minimize(cost_matrix)

        # Should assign each row to its corresponding column (zeros on diagonal)
        assert torch.equal(row_ind, torch.tensor([0, 1, 2]))
        assert torch.equal(col_ind, torch.tensor([0, 1, 2]))

    def test_scipy_comparison_maximize_square(
        self,
        greedy_matcher_maximize: GreedyLinearSumAssignment,
    ) -> None:
        """Compare with scipy on random square matrices (maximization)."""
        torch.manual_seed(42)

        # Test multiple random matrices
        agreement_count = 0
        total_tests = 100

        for _ in range(total_tests):
            cost_matrix = torch.rand(10, 10)

            # Greedy result
            row_ind_greedy, col_ind_greedy = greedy_matcher_maximize(cost_matrix)
            greedy_cost = cost_matrix[row_ind_greedy, col_ind_greedy].sum().item()

            # Scipy result
            row_ind_scipy, col_ind_scipy = scipy_linear_sum_assignment(
                cost_matrix.numpy(),
                maximize=True,
            )
            scipy_cost = cost_matrix[row_ind_scipy, col_ind_scipy].sum().item()

            # Check if greedy achieves at least 95% of optimal (scipy) cost
            if greedy_cost >= 0.95 * scipy_cost:
                agreement_count += 1

        # Greedy should achieve near-optimal results in most cases
        assert agreement_count >= 90, f"Only {agreement_count}/100 tests achieved 95% of optimal"

    def test_scipy_comparison_maximize_rectangular(
        self,
        greedy_matcher_maximize: GreedyLinearSumAssignment,
    ) -> None:
        """Compare with scipy on random rectangular matrices."""
        torch.manual_seed(123)

        for rows, cols in [(5, 10), (10, 5), (3, 8), (8, 3)]:
            cost_matrix = torch.rand(rows, cols)

            # Greedy result
            row_ind_greedy, col_ind_greedy = greedy_matcher_maximize(cost_matrix)
            greedy_cost = cost_matrix[row_ind_greedy, col_ind_greedy].sum().item()

            # Scipy result
            row_ind_scipy, col_ind_scipy = scipy_linear_sum_assignment(
                cost_matrix.numpy(),
                maximize=True,
            )
            scipy_cost = cost_matrix[row_ind_scipy, col_ind_scipy].sum().item()

            # Greedy should achieve at least 90% of optimal for rectangular matrices
            assert greedy_cost >= 0.90 * scipy_cost, (
                f"Greedy cost {greedy_cost} < 90% of scipy cost {scipy_cost} for {rows}x{cols} matrix"
            )

    def test_scipy_exact_match_small_matrices(
        self,
        greedy_matcher_maximize: GreedyLinearSumAssignment,
    ) -> None:
        """Test that greedy matches scipy exactly for many small matrices."""
        torch.manual_seed(0)

        exact_matches = 0
        total_tests = 100

        for _ in range(total_tests):
            # Small matrices where greedy often finds optimal
            cost_matrix = torch.rand(3, 3)

            row_ind_greedy, col_ind_greedy = greedy_matcher_maximize(cost_matrix)
            row_ind_scipy, col_ind_scipy = scipy_linear_sum_assignment(
                cost_matrix.numpy(),
                maximize=True,
            )

            greedy_cost = cost_matrix[row_ind_greedy, col_ind_greedy].sum().item()
            scipy_cost = cost_matrix[row_ind_scipy, col_ind_scipy].sum().item()

            if np.isclose(greedy_cost, scipy_cost, rtol=1e-5):
                exact_matches += 1

        # For 3x3 matrices, greedy should be optimal in many cases
        assert exact_matches >= 70, f"Only {exact_matches}/100 exact matches for 3x3 matrices"

    def test_empty_matrix(
        self,
        greedy_matcher_maximize: GreedyLinearSumAssignment,
    ) -> None:
        """Test with empty matrix."""
        cost_matrix = torch.empty(0, 0)

        row_ind, col_ind = greedy_matcher_maximize(cost_matrix)

        assert row_ind.numel() == 0
        assert col_ind.numel() == 0

    def test_single_element(
        self,
        greedy_matcher_maximize: GreedyLinearSumAssignment,
    ) -> None:
        """Test with single element matrix."""
        cost_matrix = torch.tensor([[5.0]])

        row_ind, col_ind = greedy_matcher_maximize(cost_matrix)

        assert torch.equal(row_ind, torch.tensor([0]))
        assert torch.equal(col_ind, torch.tensor([0]))

    def test_output_sorted_by_row(
        self,
        greedy_matcher_maximize: GreedyLinearSumAssignment,
    ) -> None:
        """Test that output is sorted by row index."""
        torch.manual_seed(999)
        cost_matrix = torch.rand(5, 5)

        row_ind, _ = greedy_matcher_maximize(cost_matrix)

        # Row indices should be sorted
        assert torch.equal(row_ind, row_ind.sort()[0])

    def test_statistical_comparison(
        self,
        greedy_matcher_maximize: GreedyLinearSumAssignment,
    ) -> None:
        """Statistical comparison of greedy vs scipy over many trials."""
        torch.manual_seed(42)

        ratios = []
        for size in [5, 10, 15, 20]:
            for _ in range(50):
                cost_matrix = torch.rand(size, size)

                row_ind_greedy, col_ind_greedy = greedy_matcher_maximize(cost_matrix)
                greedy_cost = cost_matrix[row_ind_greedy, col_ind_greedy].sum().item()

                row_ind_scipy, col_ind_scipy = scipy_linear_sum_assignment(
                    cost_matrix.numpy(),
                    maximize=True,
                )
                scipy_cost = cost_matrix[row_ind_scipy, col_ind_scipy].sum().item()

                ratios.append(greedy_cost / scipy_cost if scipy_cost > 0 else 1.0)

        mean_ratio = np.mean(ratios)
        min_ratio = np.min(ratios)

        # Average should be very close to optimal
        assert mean_ratio >= 0.95, f"Mean ratio {mean_ratio} < 0.95"
        # Worst case should still be reasonable
        assert min_ratio >= 0.85, f"Min ratio {min_ratio} < 0.85"


class TestGreedyLinearSumAssignmentComputational:
    """Computational performance and timing tests."""

    @pytest.fixture
    def greedy_matcher(self) -> GreedyLinearSumAssignment:
        """Create a greedy matcher."""
        return GreedyLinearSumAssignment(maximize=True)

    def test_performance_vs_scipy(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
    ) -> None:
        """Compare execution time with scipy."""
        torch.manual_seed(42)
        cost_matrix = torch.rand(50, 50)

        # Time greedy
        start = time.perf_counter()
        for _ in range(100):
            greedy_matcher(cost_matrix)
        greedy_time = time.perf_counter() - start

        # Time scipy
        cost_np = cost_matrix.numpy()
        start = time.perf_counter()
        for _ in range(100):
            scipy_linear_sum_assignment(cost_np, maximize=True)
        scipy_time = time.perf_counter() - start

        # Log the times instead of printing
        logger.info("Greedy time (100 iter): %.4fs", greedy_time)
        logger.info("Scipy time (100 iter): %.4fs", scipy_time)
        logger.info("Ratio (greedy/scipy): %.2fx", greedy_time / scipy_time)


class TestGreedyLinearSumAssignmentDevices:
    """Test the greedy matcher on different devices."""

    def test_cpu(self) -> None:
        """Test on CPU."""
        matcher = GreedyLinearSumAssignment(maximize=True)
        cost_matrix = torch.rand(5, 5, device="cpu")

        row_ind, col_ind = matcher(cost_matrix)

        assert row_ind.device.type == "cpu"
        assert col_ind.device.type == "cpu"
        assert row_ind.numel() == 5
        assert col_ind.numel() == 5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self) -> None:
        """Test on CUDA."""
        matcher = GreedyLinearSumAssignment(maximize=True)
        cost_matrix = torch.rand(5, 5, device="cuda")

        row_ind, col_ind = matcher(cost_matrix)

        assert row_ind.device.type == "cuda"
        assert col_ind.device.type == "cuda"
        assert row_ind.numel() == 5
        assert col_ind.numel() == 5

    @pytest.mark.skipif(not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU not available")
    def test_xpu(self) -> None:
        """Test on XPU (Intel GPU)."""
        matcher = GreedyLinearSumAssignment(maximize=True)
        cost_matrix = torch.rand(5, 5, device="xpu")

        row_ind, col_ind = matcher(cost_matrix)

        assert row_ind.device.type == "xpu"
        assert col_ind.device.type == "xpu"
        assert row_ind.numel() == 5
        assert col_ind.numel() == 5


class TestGreedyLinearSumAssignmentExport:
    """Test export compatibility with ONNX and OpenVINO."""

    @pytest.fixture
    def greedy_matcher(self) -> GreedyLinearSumAssignment:
        """Create a greedy matcher."""
        return GreedyLinearSumAssignment(maximize=True)

    def test_torch_script(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
    ) -> None:
        """Test TorchScript export."""
        scripted = torch.jit.script(greedy_matcher)

        cost_matrix = torch.rand(5, 5)
        row_ind_orig, col_ind_orig = greedy_matcher(cost_matrix)
        row_ind_script, col_ind_script = scripted(cost_matrix)

        assert torch.equal(row_ind_orig, row_ind_script)
        assert torch.equal(col_ind_orig, col_ind_script)

    def test_torch_trace(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
    ) -> None:
        """Test torch.jit.trace export."""
        cost_matrix = torch.rand(5, 5)

        # Trace the model
        traced = torch.jit.trace(greedy_matcher, cost_matrix)

        # Test with same input
        row_ind_orig, col_ind_orig = greedy_matcher(cost_matrix)
        row_ind_traced, col_ind_traced = traced(cost_matrix)

        assert torch.equal(row_ind_orig, row_ind_traced)
        assert torch.equal(col_ind_orig, col_ind_traced)

    def test_onnx_export(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
    ) -> None:
        """Test ONNX export."""
        onnx = pytest.importorskip("onnx")

        cost_matrix = torch.rand(5, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "greedy_matcher.onnx"

            # Export to ONNX
            torch.onnx.export(
                greedy_matcher,
                (cost_matrix,),
                str(onnx_path),
                input_names=["cost_matrix"],
                output_names=["row_indices", "col_indices"],
                dynamic_axes={
                    "cost_matrix": {0: "rows", 1: "cols"},
                    "row_indices": {0: "num_assignments"},
                    "col_indices": {0: "num_assignments"},
                },
                opset_version=17,
            )

            assert onnx_path.exists()

            # Validate ONNX model
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)

    def test_onnx_runtime_inference(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
    ) -> None:
        """Test ONNX Runtime inference."""
        ort = pytest.importorskip("onnxruntime")

        cost_matrix = torch.rand(5, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "greedy_matcher.onnx"

            # Export to ONNX
            torch.onnx.export(
                greedy_matcher,
                (cost_matrix,),
                str(onnx_path),
                input_names=["cost_matrix"],
                output_names=["row_indices", "col_indices"],
                opset_version=17,
            )

            # Run with ONNX Runtime
            session = ort.InferenceSession(str(onnx_path))
            ort_inputs = {"cost_matrix": cost_matrix.numpy()}
            ort_outputs = session.run(None, ort_inputs)

            row_ind_ort = torch.from_numpy(ort_outputs[0])
            col_ind_ort = torch.from_numpy(ort_outputs[1])

            # Compare with PyTorch
            row_ind_pt, col_ind_pt = greedy_matcher(cost_matrix)

            assert torch.equal(row_ind_pt, row_ind_ort)
            assert torch.equal(col_ind_pt, col_ind_ort)

    def test_openvino_export(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
    ) -> None:
        """Test OpenVINO export."""
        ov = pytest.importorskip("openvino")

        cost_matrix = torch.rand(5, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "greedy_matcher.onnx"

            # Export to ONNX first
            torch.onnx.export(
                greedy_matcher,
                (cost_matrix,),
                str(onnx_path),
                input_names=["cost_matrix"],
                output_names=["row_indices", "col_indices"],
                opset_version=17,
            )

            # Convert to OpenVINO
            core = ov.Core()
            ov_model = core.read_model(str(onnx_path))

            assert ov_model is not None
            assert len(ov_model.inputs) == 1
            assert len(ov_model.outputs) == 2

    def test_openvino_inference(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
    ) -> None:
        """Test OpenVINO inference."""
        ov = pytest.importorskip("openvino")

        cost_matrix = torch.rand(5, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "greedy_matcher.onnx"

            # Export to ONNX first
            torch.onnx.export(
                greedy_matcher,
                (cost_matrix,),
                str(onnx_path),
                input_names=["cost_matrix"],
                output_names=["row_indices", "col_indices"],
                opset_version=17,
            )

            # Run with OpenVINO
            core = ov.Core()
            compiled_model = core.compile_model(str(onnx_path), "CPU")
            infer_request = compiled_model.create_infer_request()

            infer_request.infer({0: cost_matrix.numpy()})

            row_ind_ov = torch.from_numpy(infer_request.get_output_tensor(0).data.copy())
            col_ind_ov = torch.from_numpy(infer_request.get_output_tensor(1).data.copy())

            # Compare with PyTorch
            row_ind_pt, col_ind_pt = greedy_matcher(cost_matrix)

            assert torch.equal(row_ind_pt, row_ind_ov)
            assert torch.equal(col_ind_pt, col_ind_ov)


class TestGreedyLinearSumAssignmentDtypes:
    """Test different data types."""

    @pytest.fixture
    def greedy_matcher(self) -> GreedyLinearSumAssignment:
        """Create a greedy matcher."""
        return GreedyLinearSumAssignment(maximize=True)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    def test_float_dtypes(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
        dtype: torch.dtype,
    ) -> None:
        """Test with different float dtypes."""
        cost_matrix = torch.rand(5, 5, dtype=dtype)

        row_ind, col_ind = greedy_matcher(cost_matrix)

        assert row_ind.dtype == torch.int64
        assert col_ind.dtype == torch.int64
        assert row_ind.numel() == 5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bfloat16_cuda(
        self,
        greedy_matcher: GreedyLinearSumAssignment,
    ) -> None:
        """Test with bfloat16 on CUDA."""
        cost_matrix = torch.rand(5, 5, dtype=torch.bfloat16, device="cuda")

        row_ind, col_ind = greedy_matcher(cost_matrix)

        assert row_ind.dtype == torch.int64
        assert col_ind.dtype == torch.int64
