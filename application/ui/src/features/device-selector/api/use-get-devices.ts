/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { DeviceOption } from '../device-selector.types';

// TODO: Replace with $api.useSuspenseQuery('get', '/api/v1/system/devices', ...) once the endpoint is available.
const MOCK_DEVICES: DeviceOption[] = [
    { id: 'cpu', name: 'CPU' },
    { id: 'xpu', name: 'Intel GPU' },
    { id: 'cuda', name: 'NVIDIA GPU' },
];

export const useGetDevices = (): DeviceOption[] => {
    return MOCK_DEVICES;
};
