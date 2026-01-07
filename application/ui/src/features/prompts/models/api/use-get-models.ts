/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';

import { $api, ModelListType, ModelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
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

const getDefaultModel = (id: string): ModelType => {
    return {
        id,
        config: {
            confidence_threshold: 0.38,
            model_type: 'matcher',
            num_background_points: 3,
            num_foreground_points: 5,
            precision: 'bf16',
            sam_model: 'SAM-HQ-tiny',
            encoder_model: 'dinov3_small',
            use_mask_refinement: false,
        },
        active: true,
        name: `Matcher`,
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
            createModel(getDefaultModel(uuid()));
        }
    }, [models.length, createModel]);

    return models;
};
