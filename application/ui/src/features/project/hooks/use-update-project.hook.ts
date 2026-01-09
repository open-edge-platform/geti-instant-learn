/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, type ProjectUpdateType } from '@geti-prompt/api';
import { getQueryKey } from '@geti-prompt/query-client';
import { useQueryClient } from '@tanstack/react-query';

export const useUpdateProject = () => {
    const queryClient = useQueryClient();
    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            invalidates: [['get', '/api/v1/projects']],
        },
        onSuccess: async ({ id }) => {
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
    });

    const updateProject = (id: string, body: ProjectUpdateType, onSuccess?: () => void): void => {
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
                onSuccess,
            }
        );
    };

    return {
        mutate: updateProject,
        isPending: updateProjectMutation.isPending,
    };
};
