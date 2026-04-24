/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';

import { $api, ModelListType, ModelType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { v4 as uuid } from 'uuid';

import { useCreateModel } from './use-create-model';
import { useGetSupportedModels, type SupportedModelMetadata } from './use-get-supported-models';

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

const MODEL_TYPE_DISPLAY_NAMES: Record<string, string> = {
    matcher: 'Matcher',
    perdino: 'PerDINO',
    soft_matcher: 'SoftMatcher',
    sam3: 'SAM3',
};

const toProjectModel = (meta: SupportedModelMetadata, index: number): ModelType => ({
    id: uuid(),
    config: meta.default_config,
    active: index === 0,
    name: MODEL_TYPE_DISPLAY_NAMES[meta.default_config.model_type] ?? meta.default_config.model_type,
});

export const useGetModels = () => {
    const { models } = useGetModelsQuery();
    const supportedModels = useGetSupportedModels();
    const createModel = useCreateModel();
    const hasCreatedModel = useRef(false);

    useEffect(() => {
        if (models.length === 0 && !hasCreatedModel.current) {
            hasCreatedModel.current = true;
            supportedModels.forEach((meta, index) => {
                createModel(toProjectModel(meta, index));
            });
        }
    }, [models.length, createModel, supportedModels]);

    return models;
};
