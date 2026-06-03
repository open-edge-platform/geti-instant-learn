/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { usePrefetchQuery, useSuspenseQuery } from '@tanstack/react-query';

const inferenceDevicesQueryOptions = () => {
    return $api.queryOptions('get', '/api/v1/system/devices');
};

export const usePrefetchInferenceDevices = () => {
    usePrefetchQuery(inferenceDevicesQueryOptions());
};

export const useInferenceDevices = () => {
    return useSuspenseQuery(inferenceDevicesQueryOptions());
};
