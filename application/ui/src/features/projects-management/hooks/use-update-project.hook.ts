/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, type ProjectUpdateType } from '@geti-prompt/api';

export const useUpdateProject = () => {
    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            invalidates: [['get', '/api/v1/projects']],
        },
    });

    const updateProjectName = (id: string, body: ProjectUpdateType): void => {
        updateProjectMutation.mutate({
            body,
            params: {
                path: {
                    project_id: id,
                },
            },
        });
    };

    return updateProjectName;
};
