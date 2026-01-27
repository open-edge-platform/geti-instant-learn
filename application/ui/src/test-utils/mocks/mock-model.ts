/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ModelType } from '@/api';

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
            use_nms: true,
            compile_models: true,
        },
        active: true,
        name: 'PerDINO',
        ...model,
    };
};
