/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useQueryClient } from '@tanstack/react-query';

export const useDeleteProject = () => {
    const queryClient = useQueryClient();

    const deleteProjectMutation = $api.useMutation('delete', '/api/v1/projects/{project_id}', {
        onSuccess: async () => {
            await queryClient.invalidateQueries({
                predicate: (query) => {
                    return Array.isArray(query.queryKey) && query.queryKey.includes('/api/v1/projects');
                },
            });
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
