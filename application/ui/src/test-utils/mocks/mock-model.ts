/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { MatcherModel, ModelType, Sam3Model, SupportedModelMetadataType } from '@/api';

/** Returns a mocked PerDINO model by default. Pass overrides to customize. */
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

export const getMockedMatcherModel = (model?: Partial<MatcherModel>): MatcherModel => {
    return {
        id: 'matcher-id',
        config: {
            model_type: 'matcher',
            sam_model: 'SAM-HQ-tiny',
            encoder_model: 'dinov3_small',
            precision: 'bf16',
            num_foreground_points: 5,
            num_background_points: 3,
            confidence_threshold: 0.38,
            use_mask_refinement: false,
            num_grid_cells: 8,
            preset: 'throughput',
        },
        active: true,
        name: 'Matcher',
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

export const getMockedSupportedModels = (): SupportedModelMetadataType[] => [
    {
        default_config: {
            sam_model: 'SAM-HQ-tiny',
            encoder_model: 'dinov3_small',
            precision: 'bf16',
            model_type: 'matcher',
            num_foreground_points: 5,
            num_background_points: 3,
            confidence_threshold: 0.38,
            use_mask_refinement: false,
            num_grid_cells: 8,
            preset: 'throughput',
        },
        supported_prompt_types: ['visual_polygon'],
    },
    {
        default_config: {
            sam_model: 'SAM-HQ-tiny',
            encoder_model: 'dinov3_small',
            precision: 'bf16',
            model_type: 'perdino',
            num_foreground_points: 80,
            num_background_points: 2,
            num_grid_cells: 16,
            point_selection_threshold: 0.65,
            confidence_threshold: 0.01,
        },
        supported_prompt_types: ['visual_polygon'],
    },
    {
        default_config: {
            sam_model: 'SAM-HQ-tiny',
            encoder_model: 'dinov3_small',
            precision: 'bf16',
            model_type: 'soft_matcher',
            num_foreground_points: 40,
            num_background_points: 2,
            confidence_threshold: 0.42,
            use_sampling: false,
            use_spatial_sampling: false,
            approximate_matching: false,
            softmatching_score_threshold: 0.4,
            softmatching_bidirectional: false,
        },
        supported_prompt_types: ['visual_polygon'],
    },
    {
        default_config: {
            model_type: 'sam3',
            confidence_threshold: 0.5,
            resolution: 1008,
            precision: 'fp32',
        },
        supported_prompt_types: ['text', 'visual_rectangle'],
    },
];
