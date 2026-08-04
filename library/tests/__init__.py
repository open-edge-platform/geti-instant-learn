# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests package for Geti Instant Learn."""

from instantlearn.device import DeviceInfo, DeviceType
from instantlearn.utils.constants import Backend

CPU_DEVICE = DeviceInfo(
	type=DeviceType.CPU,
	name="CPU",
	runtime_ids={Backend.TORCH: "cpu", Backend.OPENVINO: "CPU"},
)
CUDA_DEVICE = DeviceInfo(
	type=DeviceType.GPU,
	name="CUDA GPU",
	index=0,
	runtime_ids={Backend.TORCH: "cuda:0"},
)
