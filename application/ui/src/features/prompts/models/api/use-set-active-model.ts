/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { getQueryKey } from '@geti-prompt/query-client';
import { useQueryClient } from '@tanstack/react-query';

const useSetActiveModelQuery = (projectId: string) => {
    const queryClient = useQueryClient();

    return $api.useMutation('put', '/api/v1/projects/{project_id}/models/{model_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/models', { params: { path: { project_id: projectId } } }],
            ],
        },
        onSuccess: ({ id }) => {
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/v1/projects/{project_id}/models/{model_id}',
                    { params: { path: { project_id: projectId, model_id: id } } },
                ]),
            });
        },
    });
};

export const useSetActiveModel = () => {
    const { projectId } = useProjectIdentifier();
    const updateModelMutation = useSetActiveModelQuery(projectId);

    return (model: ModelType) => {
        const { id, name, config } = model;

        return updateModelMutation.mutate({
            body: { name, config, active: true },
            params: {
                path: {
                    project_id: projectId,
                    model_id: id,
                },
            },
        });
    };
};
