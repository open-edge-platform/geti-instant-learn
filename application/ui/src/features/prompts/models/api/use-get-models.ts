/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';

import { $api, MatcherModel, ModelListType, PerDINOModel, SoftMatcherModel } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { v4 as uuid } from 'uuid';

import { useCreateModel } from './use-create-model';

const useGetModelsQuery = (): ModelListType => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/models', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data;
};

const getDefaultMatcherModel = (id: string): MatcherModel => {
    return {
        id,
        config: {
            confidence_threshold: 0.38,
            model_type: 'matcher',
            num_background_points: 2,
            num_foreground_points: 40,
            precision: 'bf16',
            sam_model: 'SAM-HQ-tiny',
            encoder_model: 'dinov3_small',
            use_mask_refinement: false,
            use_nms: true,
        },
        active: false,
        name: `Matcher`,
    };
};

const getDefaultPerDINOModel = (id: string): PerDINOModel => {
    return {
        id,
        config: {
            model_type: 'perdino',
            encoder_model: 'dinov3_small',
            sam_model: 'SAM-HQ-tiny',
            num_foreground_points: 90,
            num_background_points: 2,
            num_grid_cells: 16,
            point_selection_threshold: 0.65,
            confidence_threshold: 0.42,
            precision: 'bf16',
            use_nms: true,
        },
        active: true,
        name: 'PerDINO',
    };
};

const getDefaultSoftMatcherModel = (id: string): SoftMatcherModel => {
    return {
        id,
        config: {
            model_type: 'soft_matcher',
            sam_model: 'SAM-HQ-tiny',
            encoder_model: 'dinov3_small',
            num_foreground_points: 40,
            num_background_points: 2,
            confidence_threshold: 0.42,
            use_sampling: false,
            use_spatial_sampling: false,
            approximate_matching: false,
            softmatching_score_threshold: 0.4,
            softmatching_bidirectional: false,
            precision: 'bf16',
            use_nms: true,
        },
        active: false,
        name: 'SoftMatcher',
    };
};

export const useGetModels = () => {
    const { models } = useGetModelsQuery();
    const createModel = useCreateModel();
    const hasCreatedModel = useRef(false);

    // TODO: Backend is willing to send default models soon.
    // Once that is done, we can remove this model creation logic.
    useEffect(() => {
        if (models.length === 0 && !hasCreatedModel.current) {
            hasCreatedModel.current = true;
            createModel(getDefaultPerDINOModel(uuid()));
            createModel(getDefaultSoftMatcherModel(uuid()));
            createModel(getDefaultMatcherModel(uuid()));
        }
    }, [models.length, createModel]);

    return models;
};
