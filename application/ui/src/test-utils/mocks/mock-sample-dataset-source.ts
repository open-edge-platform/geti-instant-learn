/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SampleDatasetSourceType } from '@/api';

export const getMockedSampleDatasetSource = (
    source: Partial<{ active: boolean; datasetId: string }> = {}
): SampleDatasetSourceType => {
    return {
        id: '123',
        active: source.active ?? true,
        config: {
            seekable: true,
            dataset_id: source.datasetId ?? '11111111-1111-1111-1111-111111111111',
            source_type: 'sample_dataset',
        },
    };
};
