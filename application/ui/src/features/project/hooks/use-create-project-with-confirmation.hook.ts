/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ProjectType } from '@geti-prompt/api';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../routes/paths';
import { generateUniqueProjectName } from '../utils';
import { useCreateProjectMutation } from './use-create-project-mutation.hook';

export const useCreateProjectWithConfirmation = ({
    activeProject,
    projectsNames,
}: {
    activeProject: ProjectType | undefined;
    projectsNames: string[];
}) => {
    const queryClient = useQueryClient();

    const [isCreateProjectConfirmationDialogVisible, setIsCreateProjectConfirmationDialogVisible] =
        useState<boolean>(false);

    const createProjectMutation = useCreateProjectMutation();
    const navigate = useNavigate();

    const newProjectName = generateUniqueProjectName(projectsNames);

    const invalidateActiveProject = () => {
        if (activeProject === undefined) return;

        queryClient.invalidateQueries({
            queryKey: [
                'get',
                '/api/v1/projects/{project_id}',
                {
                    params: {
                        path: {
                            project_id: activeProject.id,
                        },
                    },
                },
            ],
        });
    };

    const createProject = () => {
        const projectId = uuid();

        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    name: newProjectName,
                },
            },
            {
                onSuccess: () => {
                    if (activeProject !== undefined) {
                        closeCreateProjectConfirmationDialog();
                        invalidateActiveProject();
                    }

                    navigate(paths.project({ projectId }));
                },
            }
        );
    };

    const closeCreateProjectConfirmationDialog = () => {
        setIsCreateProjectConfirmationDialogVisible(false);
    };

    const createProjectConfirmation = () => {
        if (activeProject === undefined) {
            createProject();
        } else {
            setIsCreateProjectConfirmationDialogVisible(true);
        }
    };

    return {
        isVisible: isCreateProjectConfirmationDialogVisible,
        createProjectConfirmation,
        close: closeCreateProjectConfirmationDialog,
        createProject,
        newProjectName,
        isPending: createProjectMutation.isPending,
    };
};
