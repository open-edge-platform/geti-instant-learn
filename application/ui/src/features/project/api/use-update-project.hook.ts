/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, type ProjectUpdateType } from '@/api';
import { getQueryKey } from '@/query-client';
import { useQueryClient } from '@tanstack/react-query';

export const useUpdateProject = () => {
    const queryClient = useQueryClient();
    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            invalidates: [['get', '/api/v1/projects']],
            error: {
                notify: true,
            },
        },
    });
    const updateProject = (id: string, body: ProjectUpdateType, onSuccess?: () => Promise<void> | void): void => {
        const projectQueryKey = getQueryKey([
            'get',
            '/api/v1/projects/{project_id}',
            { params: { path: { project_id: id } } },
        ]);
        const modelsQueryKey = getQueryKey([
            'get',
            '/api/v1/projects/{project_id}/models',
            { params: { path: { project_id: id } } },
        ]);
        updateProjectMutation.mutate(
            {
                body,
                params: {
                    path: {
                        project_id: id,
                    },
                },
            },
            {
                onSuccess: async () => {
                    await queryClient.invalidateQueries({ queryKey: projectQueryKey });
                    // Invalidate models — backend may have switched the active model (e.g. on prompt_mode change)
                    await queryClient.invalidateQueries({ queryKey: modelsQueryKey });
                    await onSuccess?.();
                },
            }
        );
    };
    return {
        mutate: updateProject,
        isPending: updateProjectMutation.isPending,
    };
};
