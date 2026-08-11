# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cross-runtime device discovery and physical-device registry."""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from instantlearn.utils.constants import Backend

logger = logging.getLogger(__name__)


class DeviceType(StrEnum):
    """Physical device class, independent of a runtime's addressing scheme."""

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


class DeviceInfo(BaseModel):
    """A physical device and the exact identifiers used by each runtime."""

    model_config = ConfigDict(frozen=True)

    type: DeviceType = Field(description="Physical device type: CPU, GPU, or NPU.")
    name: str = Field(description="Human-readable device name reported by a runtime.")
    memory: int | None = Field(
        default=None,
        description="Total device memory in bytes, or null when unavailable.",
    )
    index: int | None = Field(
        default=None,
        description="Zero-based index among devices of the same type, or null for CPU.",
    )
    runtime_ids: dict[Backend, str] = Field(
        default_factory=dict,
        description="Exact device identifier used by each supported runtime.",
    )
    identity: str | None = Field(
        default=None,
        description="Internal hardware UUID or PCI identifier used to merge runtime observations.",
        exclude=True,
        repr=False,
    )

    @computed_field(description="Stable public selection key, such as 'cpu', 'gpu-0', or 'npu-0'.")
    @property
    def key(self) -> str:
        """Build a stable selection key from the physical device type and assigned index."""
        if self.type == DeviceType.CPU:
            return DeviceType.CPU.value
        if self.index is None:
            return self.type.value
        return f"{self.type.value}-{self.index}"

    def runtime_id(self, runtime: Backend) -> str | None:
        """Look up the exact identifier for ``runtime`` in this device's runtime mapping."""
        return self.runtime_ids.get(runtime)


def _normalize_identity(value: Any) -> str | None:  # noqa: ANN401
    """Normalize a hardware identifier to lowercase text without hyphens."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex()
    normalized = str(value).strip().lower().replace("-", "")
    return normalized or None


def _normalize_name(value: str) -> str:
    """Normalize a device name to lowercase alphanumeric tokens for fallback matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _torch_identity(properties: Any) -> str | None:  # noqa: ANN401
    """Read and normalize the first available UUID or PCI bus ID from Torch properties."""
    for attribute in ("uuid", "pci_bus_id"):
        identity = _normalize_identity(getattr(properties, attribute, None))
        if identity:
            return identity
    return None


def _discover_torch_family(
    torch_api: Any,  # noqa: ANN401
    device_type: DeviceType,
    prefix: str,
) -> list[DeviceInfo]:
    """Enumerate one Torch accelerator API and build runtime-specific device observations."""
    if not torch_api.is_available():
        return []

    observations: list[DeviceInfo] = []
    for index in range(torch_api.device_count()):
        properties = torch_api.get_device_properties(index)
        observations.append(
            DeviceInfo(
                type=device_type,
                name=torch_api.get_device_name(index),
                memory=getattr(properties, "total_memory", None),
                runtime_ids={Backend.TORCH: f"{prefix}:{index}"},
                identity=_torch_identity(properties),
            ),
        )
    return observations


def discover_torch_devices() -> list[DeviceInfo]:
    """Discover CPU and available XPU or CUDA devices through PyTorch APIs."""
    try:
        import torch  # noqa: PLC0415

        observations = [
            DeviceInfo(
                type=DeviceType.CPU,
                name="CPU",
                memory=None,
                runtime_ids={Backend.TORCH: "cpu"},
                identity="cpu",
            ),
        ]
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            observations.extend(_discover_torch_family(xpu, DeviceType.GPU, "xpu"))
        cuda = getattr(torch, "cuda", None)
        if cuda is not None:
            observations.extend(_discover_torch_family(cuda, DeviceType.GPU, "cuda"))
    except (ImportError, AttributeError, RuntimeError) as error:
        logger.debug("PyTorch device discovery failed: %s", error)
        return []
    return observations


def _openvino_property(core: Any, device_id: str, property_name: str) -> Any | None:  # noqa: ANN401
    """Read an OpenVINO property, returning ``None`` when the runtime cannot provide it."""
    try:
        return core.get_property(device_id, property_name)
    except (AttributeError, RuntimeError):
        return None


def discover_openvino_devices() -> list[DeviceInfo]:
    """Discover CPU, GPU, and NPU devices from OpenVINO's available-device registry."""
    try:
        import openvino as ov  # noqa: PLC0415

        core = ov.Core()
        observations: list[DeviceInfo] = []
        for runtime_id in core.available_devices:
            device_name = runtime_id.split(".", 1)[0].upper()
            try:
                device_type = DeviceType(device_name.lower())
            except ValueError:
                logger.debug("Ignoring unsupported OpenVINO device %s", runtime_id)
                continue

            full_name = _openvino_property(core, runtime_id, "FULL_DEVICE_NAME") or runtime_id
            identity = (
                "cpu"
                if device_type == DeviceType.CPU
                else _normalize_identity(_openvino_property(core, runtime_id, "DEVICE_UUID"))
            )
            observations.append(
                DeviceInfo(
                    type=device_type,
                    name=str(full_name),
                    memory=None,
                    runtime_ids={Backend.OPENVINO: runtime_id},
                    identity=identity,
                ),
            )
    except (ImportError, AttributeError, RuntimeError) as error:
        logger.debug("OpenVINO device discovery failed: %s", error)
        return []
    return observations


def _same_physical_device(left: DeviceInfo, right: DeviceInfo) -> bool:
    """Match observations by type and identity, falling back to names across runtimes."""
    if left.type != right.type:
        return False
    if left.identity and right.identity:
        return left.identity == right.identity
    if left.runtime_ids.keys() & right.runtime_ids.keys():
        return False
    return left.type != DeviceType.CPU and _normalize_name(left.name) == _normalize_name(right.name)


def merge_device_observations(observations: list[DeviceInfo]) -> list[DeviceInfo]:
    """Merge matching runtime observations, then sort and index each physical device type."""
    merged: list[DeviceInfo] = []
    for observation in observations:
        match_index = next(
            (
                index
                for index, device in enumerate(merged)
                if _same_physical_device(device, observation)
            ),
            None,
        )
        if match_index is None:
            merged.append(observation)
            continue

        current = merged[match_index]
        merged[match_index] = DeviceInfo(
            type=current.type,
            name=current.name if current.name != current.type.value.upper() else observation.name,
            memory=current.memory or observation.memory,
            runtime_ids={**current.runtime_ids, **observation.runtime_ids},
            identity=current.identity or observation.identity,
        )

    indexed: list[DeviceInfo] = []
    counters: dict[DeviceType, int] = {}
    for device in sorted(
        merged,
        key=lambda item: (item.type != DeviceType.GPU, item.type != DeviceType.NPU, item.name),
    ):
        if device.type == DeviceType.CPU:
            index = None
        else:
            index = counters.get(device.type, 0)
            counters[device.type] = index + 1
        indexed.append(
            DeviceInfo(
                type=device.type,
                name=device.name,
                memory=device.memory,
                index=index,
                runtime_ids=device.runtime_ids,
                identity=device.identity,
            ),
        )
    return indexed


def get_supported_devices() -> list[DeviceInfo]:
    """Collect Torch and OpenVINO observations and merge them into physical devices."""
    observations = [*discover_torch_devices(), *discover_openvino_devices()]
    return merge_device_observations(observations)


def get_supported_device(key: str) -> DeviceInfo:
    """Find an available physical device by its public selection key.

    Args:
        key: Public device key, such as ``cpu``, ``gpu-0``, or ``npu-0``.

    Returns:
        The discovered physical device matching ``key``.

    Raises:
        ValueError: If no available device matches ``key``.
    """
    devices = get_supported_devices()
    match = next((device for device in devices if device.key == key), None)
    if match is not None:
        return match

    available_keys = ", ".join(device.key for device in devices) or "none"
    msg = f"Device {key!r} is not available. Available devices: {available_keys}."
    raise ValueError(msg)
