# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Geti Prompt."""

import colorsys
import hashlib
import logging
import sys
from itertools import starmap
from pathlib import Path

import numpy as np
import requests
import torch
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from getiprompt.types import Masks

logger = logging.getLogger("Geti Prompt")


def setup_logger(dir_path: Path | None = None, log_level: str = "INFO") -> None:
    """Save logs to a directory and setup console logging."""
    logger = logging.getLogger("Geti Prompt")
    logger.setLevel(log_level.upper())
    logger.propagate = False  # This will prevent duplicate logs

    # Clear existing handlers to prevent duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if dir_path:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(dir_path / "logs.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s: \t%(message)s"))
    logger.addHandler(console_handler)

    # Set other loggers to a higher level to avoid verbose debug logs
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("sam2").setLevel(logging.WARNING)


def precision_to_torch_dtype(precision: str) -> torch.dtype:
    """Convert a precision string to a torch.dtype."""
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision.lower()]


def download_file(url: str, target_path: Path, sha_sum: str | None = None) -> None:
    """Download a file from a URL to a target path.

    Args:
        url: URL to download the file from
        target_path: Path to save the file to
        sha_sum: SHA-256 checksum of the file
    """
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    disable_progress = not sys.stderr.isatty()
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        " • ",
        DownloadColumn(),
        " • ",
        TransferSpeedColumn(),
        " • ",
        TimeRemainingColumn(),
        transient=True,
        disable=disable_progress,
    )

    try:  # noqa: PLR1702
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            msg = f"Downloading {target_path.name} ({total_size / (1024 * 1024):.2f} MB) from {url}..."
            logger.info(msg)

            with progress:
                task_id = progress.add_task("download", total=total_size, filename=target_path.name)
                with target_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))

            if not disable_progress and total_size > 0:
                progress.update(task_id, completed=total_size)

        if sha_sum:
            check_file_hash(target_path, sha_sum)

        msg = f"Downloaded model weights successfully to {target_path}"
        logger.info(msg)
    except Exception:
        logger.exception("An unexpected error occurred during download.")
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                msg = f"Error removing file {target_path} after error"
                logger.exception(msg)
        raise


def check_file_hash(file_path: Path, expected_hash: str) -> None:
    """Check if the file hash matches the expected hash.

    Args:
        file_path: Path to the file to check the hash of
        expected_hash: Expected SHA-256 hash of the file

    Raises:
        ValueError: If the file hash does not match the expected hash
    """
    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if file_hash != expected_hash:
        msg = f"File {file_path} has incorrect hash. Expected {expected_hash}, got {file_hash}"
        raise ValueError(msg)


def get_colors(n: int) -> np.ndarray:
    """Generate colors for a mask.

    Args:
        n: Number of colors to generate

    Returns:
        Colors for a mask
    """
    hsv_tuples = [(x / n, 0.5, 0.5) for x in range(n)]
    rgb_tuples = starmap(colorsys.hsv_to_rgb, hsv_tuples)
    return (np.array(list(rgb_tuples)) * 255).astype(np.uint8)


def masks_to_custom_masks(
    masks: list[torch.Tensor | None],
    class_id: int = 0,
) -> list[Masks]:
    """Converts torch masks to Masks objects.

    Args:
        masks: List of torch tensors with shape (N, H, W) containing masks,
                or None for samples without masks
        class_id: The class id to use for all masks

    Returns:
        List of Masks objects
    """
    mask_list = []
    for mask in masks:
        if mask is None:
            # Create empty Masks object for samples without masks
            mask_list.append(Masks())
        else:
            # mask_array has shape (N, H, W) - already binary masks per instance
            masks_obj = Masks()
            for instance_idx in range(mask.shape[0]):
                # Add each instance mask with the same class_id
                masks_obj.add(mask[instance_idx], class_id=class_id)
            mask_list.append(masks_obj)
    return mask_list
