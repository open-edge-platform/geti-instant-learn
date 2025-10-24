/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

export const useUpdateWebcamSource = () => {
    const { projectId } = useProjectIdentifier();
    const updateWebcamSourceMutation = $api.useMutation('put', '/api/v1/projects/{project_id}/sources/{source_id}', {
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

    const updateWebcamSource = (sourceId: string, deviceId: number) => {
        updateWebcamSourceMutation.mutate({
            body: {
                connected: true,
                config: {
                    source_type: 'webcam',
                    device_id: deviceId,
                },
            },
            params: {
                path: {
                    project_id: projectId,
                    source_id: sourceId,
                },
            },
        });
    };

    return {
        mutate: updateWebcamSource,
        isPending: updateWebcamSourceMutation.isPending,
    };
};
