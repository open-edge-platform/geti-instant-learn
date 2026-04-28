/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, type ProjectType, type ProjectUpdateType } from '@/api';
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

        // Snapshot for rollback
        const previousProject = queryClient.getQueryData<ProjectType>(projectQueryKey as string[]);

        // Optimistic update: merge the body into the cached project immediately
        if (previousProject) {
            queryClient.setQueryData<ProjectType>(projectQueryKey as string[], (old) => {
                if (!old) return old;

                const filtered = Object.fromEntries(Object.entries(body).filter(([, v]) => v != null));

                return { ...old, ...filtered } as ProjectType;
            });
        }

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
                    await onSuccess?.();

                    await queryClient.invalidateQueries({ queryKey: projectQueryKey });
                },
                onError: () => {
                    // Rollback on failure
                    if (previousProject) {
                        queryClient.setQueryData<ProjectType>(projectQueryKey as string[], previousProject);
                    }
                },
            }
        );
    };

    return {
        mutate: updateProject,
        isPending: updateProjectMutation.isPending,
    };
};
