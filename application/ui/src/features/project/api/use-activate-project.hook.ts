/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ProjectType } from '@geti-prompt/api';
import { getQueryKey } from '@geti-prompt/query-client';
import { useQueryClient } from '@tanstack/react-query';

import { useUpdateProject } from './use-update-project.hook';

export const useActivateProject = () => {
    const queryClient = useQueryClient();

    const updateProject = useUpdateProject();

    const activateProject = (project: ProjectType, activeProject?: ProjectType) => {
        updateProject.mutate(project.id, { active: true }, async () => {
            if (activeProject === undefined) return;

            await queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/v1/projects/{project_id}',
                    {
                        params: {
                            path: {
                                project_id: activeProject.id,
                            },
                        },
                    },
                ]),
            });
        });
    };

    return {
        mutate: activateProject,
        isPending: updateProject.isPending,
    };
};
