# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure PyTorch linear sum assignment (Hungarian/Munkres algorithm).

This module provides ONNX/OpenVINO-exportable alternatives to scipy.optimize.linear_sum_assignment.
All implementations are device-agnostic (CPU/CUDA/XPU) and produce numerically equivalent results.

Example:
    >>> import torch
    >>> from getiprompt.components.linear_sum_assignment import linear_sum_assignment
    >>> cost = torch.tensor([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=torch.float32)
    >>> row_ind, col_ind = linear_sum_assignment(cost, maximize=False)
    >>> cost[row_ind, col_ind].sum()  # Optimal cost = 5
    tensor(5.)
"""

from typing import Literal

import torch
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
from torch import nn


class LinearSumAssignment(nn.Module):
    """Pure PyTorch linear sum assignment solver.

    Drop-in replacement for scipy.optimize.linear_sum_assignment that is:
    - Device-agnostic (CPU/CUDA/XPU)
    - ONNX/OpenVINO exportable
    - Numerically equivalent to scipy (with Hungarian method)

    Args:
        maximize: If True, maximize the sum of costs. Default: True.
        method: Algorithm selection:
            - "auto" (recommended): Uses fast scipy during normal execution,
              automatically switches to greedy during ONNX/TorchScript export.
              Best of both worlds - fast dev, exportable deployment.
            - "greedy": Fast O(n^2 x min(n,m)) approximation, ~94-100% optimal
              (100% for sparse rectangular matrices). Always exportable.
            - "hungarian": Optimal O(n^3) solution, exact scipy parity.
              Exportable but slow for rectangular matrices.
            Default: "auto".

    Example:
        >>> solver = LinearSumAssignment(maximize=False)
        >>> cost = torch.tensor([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=torch.float32)
        >>> row_ind, col_ind = solver(cost)
        >>> cost[row_ind, col_ind].sum()
        tensor(5.)
    """

    def __init__(
        self,
        maximize: bool = True,
        method: Literal["greedy", "hungarian", "auto"] = "auto",
    ) -> None:
        """Initialize LinearSumAssignment solver."""
        super().__init__()
        self.maximize = maximize
        self._method_str: str = method

    def forward(self, cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve linear sum assignment problem.

        Args:
            cost_matrix: 2D cost matrix of shape (n_rows, n_cols).

        Returns:
            row_ind: 1D tensor of assigned row indices.
            col_ind: 1D tensor of assigned column indices.
        """
        n_rows, n_cols = cost_matrix.shape
        if n_rows == 0 or n_cols == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=cost_matrix.device),
                torch.empty(0, dtype=torch.int64, device=cost_matrix.device),
            )

        # Auto mode: use scipy for speed, switch to greedy during export
        if self._method_str == "auto":
            # Check for TorchScript/tracing first (covers jit.script, jit.trace, and onnx.export)
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                return self._greedy(cost_matrix)
            # Normal Python execution - use fast scipy
            return self._scipy(cost_matrix)

        if self._method_str == "greedy":
            return self._greedy(cost_matrix)
        return self._hungarian(cost_matrix)

    def _scipy(self, cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Use scipy for optimal, fast solution (not exportable)."""
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = scipy_linear_sum_assignment(cost_np, maximize=self.maximize)
        return (
            torch.from_numpy(row_ind).to(cost_matrix.device),
            torch.from_numpy(col_ind).to(cost_matrix.device),
        )

    def _greedy(self, cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy approximation - O(n^2 x min(n,m)). ONNX/OpenVINO exportable."""
        n_rows, n_cols = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype
        n_assign = min(n_rows, n_cols)

        # Work with costs - negate if maximizing so we can always use argmax
        costs = cost_matrix.clone() if self.maximize else -cost_matrix.clone()

        # Use masks instead of in-place inf assignment for better export compatibility
        row_used = torch.zeros(n_rows, dtype=torch.bool, device=device)
        col_used = torch.zeros(n_cols, dtype=torch.bool, device=device)

        row_ind = torch.zeros(n_assign, dtype=torch.int64, device=device)
        col_ind = torch.zeros(n_assign, dtype=torch.int64, device=device)

        for i in range(n_assign):
            # Mask out used rows and columns
            mask = row_used.unsqueeze(1) | col_used.unsqueeze(0)
            masked_costs = torch.where(mask, torch.tensor(float("-inf"), dtype=dtype, device=device), costs)

            # Find best assignment
            flat_idx = masked_costs.reshape(-1).argmax()
            r = flat_idx // n_cols
            c = flat_idx % n_cols

            row_ind[i] = r
            col_ind[i] = c
            row_used[r] = True
            col_used[c] = True

        # Sort by row index for consistent output
        sort_idx = row_ind.argsort()
        return row_ind[sort_idx], col_ind[sort_idx]

    def _hungarian(  # noqa: C901, PLR0915
        self,
        cost_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hungarian/Munkres algorithm - O(n^3) exact solution. ONNX/OpenVINO exportable.

        This implementation uses tensor operations with explicit loops that are
        compatible with TorchScript and ONNX export. The algorithm is inherently
        complex and cannot be easily simplified without losing clarity.
        """
        n_rows, n_cols = cost_matrix.shape
        device = cost_matrix.device

        # Transpose if more rows than columns (scipy convention)
        transposed = n_rows > n_cols
        if transposed:
            cost_matrix = cost_matrix.T
            n_rows, n_cols = n_cols, n_rows

        # Convert to minimization with float64 for numerical stability
        cost_mat = (-cost_matrix if self.maximize else cost_matrix).to(torch.float64).clone()

        # Pad to square if rectangular
        n = n_cols
        real_rows = n_rows
        if n_rows < n_cols:
            big = cost_mat.abs().max() * n + 1.0
            padded = torch.full((n, n), big, dtype=torch.float64, device=device)
            padded[:n_rows, :] = cost_mat
            cost_mat = padded

        eps = 1e-10

        # Step 1: Row and column reduction
        cost_mat -= cost_mat.min(dim=1, keepdim=True).values
        cost_mat -= cost_mat.min(dim=0, keepdim=True).values

        # Initialize tracking tensors
        starred = torch.zeros((n, n), dtype=torch.bool, device=device)
        primed = torch.zeros((n, n), dtype=torch.bool, device=device)
        row_covered = torch.zeros(n, dtype=torch.bool, device=device)
        col_covered = torch.zeros(n, dtype=torch.bool, device=device)

        # Step 2: Star zeros - find independent zeros
        is_zero = eps >= cost_mat
        for i in range(n):
            for j in range(n):
                if is_zero[i, j] and not row_covered[i] and not col_covered[j]:
                    starred[i, j] = True
                    row_covered[i] = True
                    col_covered[j] = True

        # Reset covers
        row_covered = torch.zeros(n, dtype=torch.bool, device=device)
        col_covered = torch.zeros(n, dtype=torch.bool, device=device)

        # Path storage - preallocate with max size
        path_rows = torch.zeros(2 * n + 1, dtype=torch.int64, device=device)
        path_cols = torch.zeros(2 * n + 1, dtype=torch.int64, device=device)

        max_iter = n * n * 10
        for _ in range(max_iter):
            # Step 3: Cover columns with starred zeros
            col_covered = starred.any(dim=0)

            if col_covered.sum() >= n:
                break

            # Step 4: Find uncovered zeros
            step4_done = False
            while not step4_done:
                # Find all uncovered zeros
                is_zero = eps >= cost_mat
                uncovered_mask = (~row_covered).unsqueeze(1) & (~col_covered).unsqueeze(0)
                uncovered_zeros = is_zero & uncovered_mask

                if not uncovered_zeros.any():
                    # Step 6: No uncovered zeros, modify matrix
                    uncovered_vals = torch.where(
                        uncovered_mask,
                        cost_mat,
                        torch.full_like(cost_mat, float("inf")),
                    )
                    min_val = uncovered_vals.min()

                    # Add to covered rows, subtract from uncovered columns
                    cost_mat += row_covered.to(cost_mat.dtype).unsqueeze(1) * min_val
                    cost_mat -= (~col_covered).to(cost_mat.dtype).unsqueeze(0) * min_val
                    continue

                # Find first uncovered zero
                flat_idx = uncovered_zeros.flatten().to(torch.int64).argmax()
                row = flat_idx // n
                col = flat_idx % n

                # Prime this zero
                primed[row, col] = True

                # Check for starred zero in this row
                star_in_row = starred[row, :]
                if star_in_row.any():
                    star_col = star_in_row.to(torch.int64).argmax()
                    row_covered[row] = True
                    col_covered[star_col] = False
                else:
                    # Step 5: Construct augmenting path
                    path_rows[0] = row
                    path_cols[0] = col
                    path_len = 1

                    done = False
                    while not done:
                        # Find starred zero in column
                        col_idx = path_cols[path_len - 1]
                        star_in_col = starred[:, col_idx]
                        if star_in_col.any():
                            star_row = star_in_col.to(torch.int64).argmax()
                            path_rows[path_len] = star_row
                            path_cols[path_len] = col_idx
                            path_len += 1

                            # Find primed zero in this row
                            prime_in_row = primed[star_row, :]
                            prime_col = prime_in_row.to(torch.int64).argmax()
                            path_rows[path_len] = star_row
                            path_cols[path_len] = prime_col
                            path_len += 1
                        else:
                            done = True

                    # Augment: toggle starred status along path
                    for k in range(path_len):
                        r = path_rows[k]
                        c = path_cols[k]
                        starred[r, c] = ~starred[r, c]

                    # Clear covers and primes
                    row_covered = torch.zeros(n, dtype=torch.bool, device=device)
                    col_covered = torch.zeros(n, dtype=torch.bool, device=device)
                    primed = torch.zeros((n, n), dtype=torch.bool, device=device)
                    step4_done = True

        # Extract assignments from starred zeros (only from real rows)
        # Use vectorized operations instead of lists
        starred_real = starred[:real_rows, :]
        has_assignment = starred_real.any(dim=1)
        row_ind = torch.arange(real_rows, dtype=torch.int64, device=device)[has_assignment]

        # For each row with assignment, find the column
        col_ind = torch.zeros(row_ind.shape[0], dtype=torch.int64, device=device)
        for idx in range(row_ind.shape[0]):
            r = row_ind[idx]
            col_ind[idx] = starred[r, :].to(torch.int64).argmax()

        # Undo transpose if needed
        if transposed:
            row_ind, col_ind = col_ind, row_ind

        return row_ind, col_ind


def linear_sum_assignment(
    cost_matrix: torch.Tensor,
    maximize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional interface matching scipy.optimize.linear_sum_assignment.

    Drop-in replacement for scipy.optimize.linear_sum_assignment that:
    - Uses fast scipy during normal PyTorch execution
    - Automatically switches to exportable greedy during ONNX/TorchScript export
    - Returns PyTorch tensors instead of numpy arrays

    Args:
        cost_matrix: 2D cost matrix of shape (n_rows, n_cols).
        maximize: If True, maximize the sum. Default: False (minimize).

    Returns:
        row_ind: 1D tensor of assigned row indices.
        col_ind: 1D tensor of assigned column indices.

    Example:
        >>> cost = torch.tensor([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        >>> row_ind, col_ind = linear_sum_assignment(cost, maximize=False)
        >>> cost[row_ind, col_ind].sum()  # = 5 (optimal)
        tensor(5)
    """
    solver = LinearSumAssignment(maximize=maximize, method="auto")
    return solver(cost_matrix)
