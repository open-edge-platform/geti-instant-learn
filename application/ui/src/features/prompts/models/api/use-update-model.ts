/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { getQueryKey } from '@/query-client';
import { useQueryClient } from '@tanstack/react-query';

const useUpdateModelMutation = (projectId: string) => {
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

export const useUpdateModel = () => {
    const { projectId } = useProjectIdentifier();
    const updateModelMutation = useUpdateModelMutation(projectId);

    const updateModel = (model: ModelType, onSuccess?: () => void) => {
        const { id, name, config, active } = model;

        return updateModelMutation.mutate(
            {
                body: {
                    name,
                    config,
                    active,
                },
                params: {
                    path: {
                        project_id: projectId,
                        model_id: id,
                    },
                },
            },
            {
                onSuccess,
            }
        );
    };

    return {
        mutate: updateModel,
        isPending: updateModelMutation.isPending,
    };
};
