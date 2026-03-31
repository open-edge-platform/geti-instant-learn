/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ModelType, Sam3Model } from '@/api';

export const getMockedModel = (model?: Partial<ModelType>): ModelType => {
    return {
        id: 'some-id',
        config: {
            model_type: 'perdino',
            encoder_model: 'dinov3_large',
            sam_model: 'SAM-HQ-tiny',
            num_foreground_points: 40,
            num_background_points: 2,
            num_grid_cells: 16,
            point_selection_threshold: 0.65,
            confidence_threshold: 0.42,
            precision: 'bf16',
        },
        active: true,
        name: 'PerDINO',
        ...model,
    };
};

export const getMockedSam3Model = (model?: Partial<Sam3Model>): Sam3Model => {
    return {
        id: 'sam3-id',
        config: {
            model_type: 'sam3',
            confidence_threshold: 0.5,
            resolution: 1008,
            precision: 'fp32',
        },
        active: false,
        name: 'SAM3',
        ...model,
    };
};
