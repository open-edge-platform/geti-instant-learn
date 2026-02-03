/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';

export const useDeleteProject = () => {
    const deleteProjectMutation = $api.useMutation('delete', '/api/v1/projects/{project_id}', {
        meta: {
            awaits: [['get', '/api/v1/projects']],
            error: {
                notify: true,
            },
        },
    });

    const deleteProject = (id: string, onSuccess?: () => void): void => {
        deleteProjectMutation.mutate(
            {
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

    return deleteProject;
};
