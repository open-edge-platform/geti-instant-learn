/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';

export const useUpdateProject = () => {
    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}');

    const updateProjectName = (id: string, name: string): void => {
        updateProjectMutation.mutate({
            body: {
                name,
            },
            params: {
                path: {
                    project_id: id,
                },
            },
        });
    };

    return updateProjectName;
};
