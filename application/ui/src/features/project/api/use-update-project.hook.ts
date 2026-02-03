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

                    await queryClient.invalidateQueries({
                        queryKey: getQueryKey([
                            'get',
                            '/api/v1/projects/{project_id}',
                            {
                                params: {
                                    path: {
                                        project_id: id,
                                    },
                                },
                            },
                        ]),
                    });
                },
            }
        );
    };

    return {
        mutate: updateProject,
        isPending: updateProjectMutation.isPending,
    };
};
