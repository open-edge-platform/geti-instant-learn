/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';

export const useDeleteSource = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('delete', '/api/v1/projects/{project_id}/sources/{source_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/sources', { params: { path: { project_id: projectId } } }],
            ],
            error: {
                notify: true,
            },
        },
    });
};
