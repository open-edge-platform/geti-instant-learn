/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';

const useCreateLabelMutation = (projectId: string) => {
    return $api.useMutation('post', '/api/v1/projects/{project_id}/labels', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/labels', { params: { path: { project_id: projectId } } }],
            ],
            error: {
                notify: true,
            },
        },
    });
};

export const useCreateLabel = () => {
    const { projectId } = useProjectIdentifier();

    return useCreateLabelMutation(projectId);
};
