/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SourceConfig } from '@/api';
import { useProjectIdentifier } from '@/hooks';

export const useUpdateSource = (sourceId: string) => {
    const { projectId } = useProjectIdentifier();

    const updateSourceMutation = $api.useMutation('put', '/api/v1/projects/{project_id}/sources/{source_id}', {
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
                [
                    'get',
                    '/api/v1/projects/{project_id}/sources/{source_id}/frames',
                    {
                        params: {
                            path: {
                                project_id: projectId,
                                source_id: sourceId,
                            },
                        },
                    },
                ],
            ],
            error: {
                notify: true,
            },
        },
    });

    const updateSource = (body: { config: SourceConfig; active: boolean }, onSuccess?: () => void) => {
        updateSourceMutation.mutate(
            {
                body,
                params: {
                    path: {
                        project_id: projectId,
                        source_id: sourceId,
                    },
                },
            },
            {
                onSuccess,
            }
        );
    };

    return {
        mutate: updateSource,
        isPending: updateSourceMutation.isPending,
    };
};
