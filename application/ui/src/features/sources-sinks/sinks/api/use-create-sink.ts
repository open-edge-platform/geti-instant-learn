/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SinkCreateType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { v4 as uuid } from 'uuid';

const useCreateSinkMutation = (projectId: string) => {
    return $api.useMutation('post', '/api/v1/projects/{project_id}/sinks', {
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

export const useCreateSink = () => {
    const { projectId } = useProjectIdentifier();
    const createSinkMutation = useCreateSinkMutation(projectId);

    const createSink = (config: SinkCreateType['config'], onSuccess: () => void) => {
        createSinkMutation.mutate(
            {
                body: {
                    id: uuid(),
                    active: true,
                    config,
                },
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            },
            {
                onSuccess,
            }
        );
    };

    return {
        mutate: createSink,
        isPending: createSinkMutation.isPending,
    };
};
