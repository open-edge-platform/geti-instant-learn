/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

const useDeleteSinkMutation = (projectId: string) => {
    return $api.useMutation('delete', '/api/v1/projects/{project_id}/sinks/{sink_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/sinks', { params: { path: { project_id: projectId } } }],
            ],
            error: {
                notify: true,
            },
        },
    });
};

export const useDeleteSink = () => {
    const { projectId } = useProjectIdentifier();

    return useDeleteSinkMutation(projectId);
};
