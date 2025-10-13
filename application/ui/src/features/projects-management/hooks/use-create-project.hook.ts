/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../routes/paths';

export const useCreateProject = () => {
    const createProjectMutation = $api.useMutation('post', '/api/v1/projects');
    const navigate = useNavigate();

    const createProject = (projectName: string, projectId: string = uuid()) => {
        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    name: projectName,
                },
            },
            {
                onSuccess: () => {
                    navigate(paths.project({ projectId }));
                },
            }
        );
    };

    return createProject;
};
