/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';

const useDeleteLabelMutation = (projectId: string) => {
    return $api.useMutation('delete', '/api/v1/projects/{project_id}/labels/{label_id}', {
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

export const useDeleteLabel = () => {
    const { projectId } = useProjectIdentifier();

    return useDeleteLabelMutation(projectId);
};
