# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.project import Device


def has_intel_gpu() -> bool:
    """Check whether an Intel GPU backend is available via PyTorch XPU."""
    try:
        import torch

        return torch.xpu.is_available()
    except (ImportError, AttributeError, RuntimeError):
        return False


def has_nvidia_gpu() -> bool:
    """Check whether a CUDA-capable NVIDIA GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except (ImportError, AttributeError, RuntimeError):
        return False


def list_available_devices() -> list[Device]:
    """List all currently available runtime devices.

    CPU is always available. Intel XPU and NVIDIA CUDA are added when detected.
    """
    devices: list[Device] = []
    if has_intel_gpu():
        devices.append(Device.XPU)
    if has_nvidia_gpu():
        devices.append(Device.CUDA)
    devices.append(Device.CPU)
    return devices
