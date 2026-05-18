/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, type ProjectUpdateType } from '@/api';

export const useUpdateProject = () => {
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
                },
            }
        );
    };

    return {
        mutate: updateProject,
        isPending: updateProjectMutation.isPending,
    };
};
