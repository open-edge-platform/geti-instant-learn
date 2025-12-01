/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SinkUpdateType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

const useUpdateSinkMutation = (projectId: string) => {
    return $api.useMutation('put', '/api/v1/projects/{project_id}/sinks/{sink_id}', {
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

export const useUpdateSink = () => {
    const { projectId } = useProjectIdentifier();
    const updateSinkMutation = useUpdateSinkMutation(projectId);

    const updateSink = ({ sinkId, body }: { sinkId: string; body: SinkUpdateType }, onSuccess: () => void) => {
        updateSinkMutation.mutate(
            {
                body,
                params: {
                    path: {
                        project_id: projectId,
                        sink_id: sinkId,
                    },
                },
            },
            {
                onSuccess,
            }
        );
    };

    return {
        mutate: updateSink,
        isPending: updateSinkMutation.isPending,
    };
};
