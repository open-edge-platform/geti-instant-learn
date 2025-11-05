/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { v4 as uuid } from 'uuid';

export const useCreateWebcamSource = () => {
    const { projectId } = useProjectIdentifier();
    const createWebcamSourceMutation = $api.useMutation('post', '/api/v1/projects/{project_id}/sources', {
        meta: {
            invalidates: [
                [
                    'get',
                    '/api/v1/projects/{project_id}/sources',
                    {
                        params: {
                            path: {
                                project_id: projectId,
                            },
                        },
                    },
                ],
            ],
        },
    });

    const createWebcamSource = (deviceId: number) => {
        createWebcamSourceMutation.mutate({
            body: {
                id: uuid(),
                connected: true,
                config: {
                    seekable: false,
                    source_type: 'webcam',
                    device_id: deviceId,
                },
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    return {
        mutate: createWebcamSource,
        isPending: createWebcamSourceMutation.isPending,
    };
};
