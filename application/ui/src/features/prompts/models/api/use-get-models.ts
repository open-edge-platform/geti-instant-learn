/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';

import { $api, ModelListType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

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

export const useGetModels = () => {
    const { models } = useGetModelsQuery();
    const createModel = useCreateModel();
    const hasCreatedModel = useRef(false);

    // TODO: Backend is willing to send default models soon.
    // Once that is done, we can remove this model creation logic.
    useEffect(() => {
        if (models.length === 0 && !hasCreatedModel.current) {
            hasCreatedModel.current = true;
            createModel();
        }
    }, [models.length, createModel]);

    return models;
};
