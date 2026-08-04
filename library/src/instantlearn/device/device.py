# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cross-runtime device discovery and physical-device registry."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from instantlearn.utils.constants import Backend

logger = logging.getLogger(__name__)


class DeviceType(StrEnum):
    """Physical device class, independent of a runtime's addressing scheme."""

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


@dataclass(frozen=True)
class DeviceInfo:
    """A physical device and the exact identifiers used by each runtime."""

    type: DeviceType
    name: str
    memory: int | None = None
    index: int | None = None
    runtime_ids: dict[Backend, str] = field(default_factory=dict)
    identity: str | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        """Return the stable selection key exposed to application callers."""
        if self.type == DeviceType.CPU:
            return DeviceType.CPU.value
        if self.index is None:
            return self.type.value
        return f"{self.type.value}-{self.index}"

    def runtime_id(self, runtime: Backend) -> str | None:
        """Return this device's identifier for ``runtime``, if available."""
        return self.runtime_ids.get(runtime)


@dataclass(frozen=True)
class _DeviceObservation:
    type: DeviceType
    name: str
    memory: int | None
    runtime: Backend
    runtime_id: str
    identity: str | None = None


def _normalize_identity(value: Any) -> str | None:  # noqa: ANN401
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex()
    normalized = str(value).strip().lower().replace("-", "")
    return normalized or None


def _normalize_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _torch_identity(properties: Any) -> str | None:  # noqa: ANN401
    for attribute in ("uuid", "pci_bus_id"):
        identity = _normalize_identity(getattr(properties, attribute, None))
        if identity:
            return identity
    return None


def _discover_torch_family(
    torch_api: Any,  # noqa: ANN401
    device_type: DeviceType,
    prefix: str,
) -> list[_DeviceObservation]:
    if not torch_api.is_available():
        return []

    observations: list[_DeviceObservation] = []
    for index in range(torch_api.device_count()):
        properties = torch_api.get_device_properties(index)
        observations.append(
            _DeviceObservation(
                type=device_type,
                name=torch_api.get_device_name(index),
                memory=getattr(properties, "total_memory", None),
                runtime=Backend.TORCH,
                runtime_id=f"{prefix}:{index}",
                identity=_torch_identity(properties),
            ),
        )
    return observations


def discover_torch_devices() -> list[_DeviceObservation]:
    """Discover devices addressable by PyTorch."""
    try:
        import torch  # noqa: PLC0415

        observations = [
            _DeviceObservation(
                type=DeviceType.CPU,
                name="CPU",
                memory=None,
                runtime=Backend.TORCH,
                runtime_id="cpu",
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
    try:
        return core.get_property(device_id, property_name)
    except (AttributeError, RuntimeError):
        return None


def discover_openvino_devices() -> list[_DeviceObservation]:
    """Discover devices addressable by OpenVINO."""
    try:
        import openvino as ov  # noqa: PLC0415

        core = ov.Core()
        observations: list[_DeviceObservation] = []
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
                _DeviceObservation(
                    type=device_type,
                    name=str(full_name),
                    memory=None,
                    runtime=Backend.OPENVINO,
                    runtime_id=runtime_id,
                    identity=identity,
                ),
            )
    except (ImportError, AttributeError, RuntimeError) as error:
        logger.debug("OpenVINO device discovery failed: %s", error)
        return []
    return observations


def _same_physical_device(left: _DeviceObservation, right: _DeviceObservation) -> bool:
    if left.type != right.type:
        return False
    if left.identity and right.identity:
        return left.identity == right.identity
    if left.runtime == right.runtime:
        return False
    return left.type != DeviceType.CPU and _normalize_name(left.name) == _normalize_name(right.name)


def merge_device_observations(observations: list[_DeviceObservation]) -> list[DeviceInfo]:
    """Merge runtime observations that identify the same physical device."""
    merged: list[DeviceInfo] = []
    for observation in observations:
        match_index = next(
            (
                index
                for index, device in enumerate(merged)
                if _same_physical_device(
                    _DeviceObservation(
                        type=device.type,
                        name=device.name,
                        memory=device.memory,
                        runtime=next(iter(device.runtime_ids)),
                        runtime_id=next(iter(device.runtime_ids.values())),
                        identity=device.identity,
                    ),
                    observation,
                )
            ),
            None,
        )
        if match_index is None:
            merged.append(
                DeviceInfo(
                    type=observation.type,
                    name=observation.name,
                    memory=observation.memory,
                    runtime_ids={observation.runtime: observation.runtime_id},
                    identity=observation.identity,
                ),
            )
            continue

        current = merged[match_index]
        merged[match_index] = DeviceInfo(
            type=current.type,
            name=current.name if current.name != current.type.value.upper() else observation.name,
            memory=current.memory or observation.memory,
            runtime_ids={**current.runtime_ids, observation.runtime: observation.runtime_id},
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


def enumerate_system_devices() -> list[DeviceInfo]:
    """Enumerate physical devices available through supported runtimes."""
    observations = [*discover_torch_devices(), *discover_openvino_devices()]
    return merge_device_observations(observations)
