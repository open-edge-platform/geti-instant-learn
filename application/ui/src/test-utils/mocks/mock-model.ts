/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelType } from '@geti-prompt/api';

export const getMockedModel = (model?: Partial<ModelType>): ModelType => {
    return {
        id: 'some-id',
        config: {
            mask_similarity_threshold: 0.38,
            model_type: 'matcher',
            num_background_points: 2,
            num_foreground_points: 40,
            precision: 'bf16',
        },
        active: true,
        name: 'Mega model',
        ...model,
    };
};
