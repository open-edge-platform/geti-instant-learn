# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Geti Prompt."""

import hashlib
import logging
import sys
from pathlib import Path

import openvino as ov
import requests
import torch
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

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
    """Convert a precision string to a torch.dtype.

    Args:
        precision: Precision name ("fp32", "fp16", "bf16")

    Returns:
        Corresponding torch.dtype

    Raises:
        ValueError: If precision is not supported
    """
    precision_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    precision_lower = precision.lower()
    if precision_lower not in precision_map:
        msg = f"Unsupported precision: {precision}. Supported values: {list(precision_map.keys())}"
        raise ValueError(msg)
    return precision_map[precision_lower]


def setup_autocast(device: str, precision: torch.dtype) -> torch.autocast:
    """Setup autocast context based on device and precision.

    Args:
        device: The device to use ('cuda', 'xpu', or 'cpu').
        precision: The torch.dtype precision (e.g., torch.bfloat16, torch.float32).

    Returns:
        Autocast context manager.
    """
    if device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        device_type = "xpu"
        supports_bf16 = precision == torch.bfloat16
    elif device == "cuda" and torch.cuda.is_available():
        device_type = "cuda"
        supports_bf16 = precision == torch.bfloat16
    else:
        device_type = "cpu"
        supports_bf16 = False

    if supports_bf16:
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device_type, dtype=torch.float32)


def precision_to_openvino_type(precision: str) -> ov.Type:
    """Convert a precision string to an OpenVINO Type.

    Args:
        precision: Precision name in PyTorch style ("fp32", "fp16", "bf16")

    Returns:
        OpenVINO Type (ov.Type.f32, ov.Type.f16, etc.)

    Raises:
        ImportError: If openvino is not installed
        ValueError: If precision is not supported
    """
    precision_map = {
        "fp32": ov.Type.f32,
        "fp16": ov.Type.f16,
        "bf16": ov.Type.f16,
    }
    precision_lower = precision.lower()
    if precision_lower not in precision_map:
        msg = f"Unsupported precision: {precision}. Supported values: {list(precision_map.keys())}"
        raise ValueError(msg)
    return precision_map[precision_lower]


def device_to_openvino_device(device: str) -> str:
    """Map PyTorch device names to OpenVINO device names.

    Args:
        device: Device name in PyTorch style ("cuda", "cpu") or OpenVINO style ("GPU", "CPU", "AUTO")

    Returns:
        Device name in OpenVINO style ("CPU", "GPU", "AUTO")
    """
    if not device:
        return "CPU"
    device_upper = device.upper()
    # Map PyTorch-style names to OpenVINO names
    if device_upper == "CUDA":
        return "GPU"
    # OpenVINO names pass through unchanged
    if device_upper in {"CPU", "GPU", "AUTO"}:
        return device_upper
    # Default fallback
    return "CPU"


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
