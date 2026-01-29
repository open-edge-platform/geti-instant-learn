/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../constants/paths';

export const useCreateProjectMutation = () => {
    return $api.useMutation('post', '/api/v1/projects', {
        meta: {
            invalidates: [['get', '/api/v1/projects']],
            error: {
                notify: true,
            },
        },
    });
};

export const useCreateProject = () => {
    const navigate = useNavigate();
    const createProjectMutation = useCreateProjectMutation();

    const createProject = ({ id, name }: { id?: string; name: string }) => {
        const projectId = id ?? uuid();

        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    name,
                },
            },
            {
                onSuccess: () => {
                    navigate(paths.project({ projectId }));
                },
            }
        );
    };

    return {
        mutate: createProject,
        isPending: createProjectMutation.isPending,
    };
};
