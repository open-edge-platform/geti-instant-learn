/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, type ProjectUpdateType } from '@geti-prompt/api';
import { useQueryClient } from '@tanstack/react-query';

export const useUpdateProject = () => {
    const queryClient = useQueryClient();
    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            invalidates: [['get', '/api/v1/projects']],
        },
    });

    const updateProjectName = (id: string, body: ProjectUpdateType): void => {
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
                onSuccess: () => {
                    queryClient.invalidateQueries({
                        queryKey: [
                            'get',
                            '/api/v1/projects/{project_id}',
                            {
                                params: {
                                    path: {
                                        project_id: id,
                                    },
                                },
                            },
                        ],
                    });
                },
            }
        );
    };

    return updateProjectName;
};
