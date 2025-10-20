/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api } from '@geti-prompt/api';

import { MutationMeta } from '../../../query-client/query-client.interface';

export const useProjectActivityManagement = (newProjectActiveId: string, currentActiveProjectId?: string) => {
    const [isProjectActiveDialogOpen, setIsProjectActiveDialogOpen] = useState<boolean>(false);

    const invalidates: MutationMeta['invalidates'] =
        currentActiveProjectId === undefined
            ? [
                  ['get', '/api/v1/projects'],
                  ['get', '/api/v1/projects/active'],
                  ['get', '/api/v1/projects/{project_id}', { params: { path: { project_id: newProjectActiveId } } }],
              ]
            : [
                  ['get', '/api/v1/projects'],
                  ['get', '/api/v1/projects/active'],
                  [
                      'get',
                      '/api/v1/projects/{project_id}',
                      { params: { path: { project_id: currentActiveProjectId } } },
                  ],
                  ['get', '/api/v1/projects/{project_id}', { params: { path: { project_id: newProjectActiveId } } }],
              ];

    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            invalidates,
        },
        onSuccess: () => {
            handleCloseProjectActiveDialog();
        },
    });

    const handleCloseProjectActiveDialog = () => {
        setIsProjectActiveDialogOpen(false);
    };

    const handleUpdateProjectActivityStatus = (isGoingToBeActive: boolean) => {
        updateProjectMutation.mutate({
            body: {
                active: isGoingToBeActive,
            },
            params: {
                path: {
                    project_id: newProjectActiveId,
                },
            },
        });
    };

    const handleDeactivateProject = () => {
        handleUpdateProjectActivityStatus(false);
    };

    const handleActivateProject = () => {
        handleUpdateProjectActivityStatus(true);
    };

    const handleShowActivateProjectDialog = () => {
        setIsProjectActiveDialogOpen(true);
    };

    return {
        isVisible: isProjectActiveDialogOpen,
        onClose: handleCloseProjectActiveDialog,
        onDeactivate: handleDeactivateProject,
        onActivate: handleActivateProject,
        onShowActivateProjectDialog: handleShowActivateProjectDialog,
        isPending: updateProjectMutation.isPending,
    };
};
