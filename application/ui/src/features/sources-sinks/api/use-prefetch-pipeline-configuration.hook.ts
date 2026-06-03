/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { usePrefetchInferenceDevices } from '../inference-device/api/use-inference-devices';
import { usePrefetchSinks } from '../sinks/api/use-sinks';
import { usePrefetchSources } from '../sources/api/use-available-sources';

export const usePrefetchPipelineConfiguration = () => {
    usePrefetchInferenceDevices();
    usePrefetchSources();
    usePrefetchSinks();
};
