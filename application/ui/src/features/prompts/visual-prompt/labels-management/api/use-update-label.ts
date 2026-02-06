/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';

const useUpdateLabelMutation = (projectId: string) => {
    return $api.useMutation('put', '/api/v1/projects/{project_id}/labels/{label_id}', {
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

export const useUpdateLabel = () => {
    const { projectId } = useProjectIdentifier();

    return useUpdateLabelMutation(projectId);
};
