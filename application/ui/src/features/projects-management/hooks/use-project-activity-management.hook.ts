/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api } from '@geti-prompt/api';

export const useProjectActivityManagement = (projectId: string) => {
    const [isProjectActiveDialogOpen, setIsProjectActiveDialogOpen] = useState<boolean>(false);

    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects'],
                ['get', '/api/v1/projects/active'],
                ['get', '/api/v1/projects/{project_id}', { params: { path: { project_id: projectId } } }],
            ],
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
                    project_id: projectId,
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
