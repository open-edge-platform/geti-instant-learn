/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api } from '@geti-prompt/api';
import { getQueryKey } from '@geti-prompt/query-client';

const useProjectActivationMutation = (projectId: string, currentActiveProjectId: string | undefined) => {
    const invalidates =
        currentActiveProjectId === undefined
            ? [
                  getQueryKey(['get', '/api/v1/projects']),
                  getQueryKey(['get', '/api/v1/projects/active']),
                  getQueryKey([
                      'get',
                      '/api/v1/projects/{project_id}',
                      { params: { path: { project_id: projectId } } },
                  ]),
              ]
            : [
                  getQueryKey(['get', '/api/v1/projects']),
                  getQueryKey(['get', '/api/v1/projects/active']),
                  getQueryKey([
                      'get',
                      '/api/v1/projects/{project_id}',
                      { params: { path: { project_id: currentActiveProjectId } } },
                  ]),
                  getQueryKey([
                      'get',
                      '/api/v1/projects/{project_id}',
                      { params: { path: { project_id: projectId } } },
                  ]),
              ];

    return $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: { invalidates },
    });
};

export const useProjectActivityManagement = (
    newActiveProjectId: string,
    currentActiveProjectId: string | undefined
) => {
    const [isProjectActiveDialogOpen, setIsProjectActiveDialogOpen] = useState<boolean>(false);

    const updateProjectMutation = useProjectActivationMutation(newActiveProjectId, currentActiveProjectId);

    const closeProjectActiveDialog = () => {
        setIsProjectActiveDialogOpen(false);
    };

    const updateProjectActivityStatus = (isGoingToBeActive: boolean) => {
        updateProjectMutation.mutate(
            {
                body: {
                    active: isGoingToBeActive,
                },
                params: {
                    path: {
                        project_id: newActiveProjectId,
                    },
                },
            },
            {
                onSuccess: () => {
                    closeProjectActiveDialog();
                },
            }
        );
    };

    const deactivateProject = () => {
        updateProjectActivityStatus(false);
    };

    const activateProject = () => {
        // If there is no active project, we just activate the selected project directly.
        if (currentActiveProjectId === undefined) {
            updateProjectActivityStatus(true);
        } else {
            showActivateProjectDialog();
        }
    };

    const showActivateProjectDialog = () => {
        setIsProjectActiveDialogOpen(true);
    };

    return {
        isVisible: isProjectActiveDialogOpen,
        close: closeProjectActiveDialog,
        deactivate: deactivateProject,
        activate: activateProject,
        activateConfirmation: () => updateProjectActivityStatus(true),
        isPending: updateProjectMutation.isPending,
    };
};
