/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';

import { queryClient } from '../../../query-client';

export const useDeleteProject = () => {
    const deleteProjectMutation = $api.useMutation('delete', '/api/v1/projects/{project_id}', {
        onSettled: async () => {
            await queryClient.invalidateQueries({
                predicate: (query) => Array.isArray(query.queryKey) && query.queryKey.includes('/api/v1/projects'),
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
