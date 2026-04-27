/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SourceConfig } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { getQueryKey } from '@/query-client';
import { useQueryClient } from '@tanstack/react-query';

/**
 * Hook for updating/connecting sources.
 *
 * @example
 * const updateSource = useUpdateSource();
 * updateSource.mutate({ sourceId, config, active }, onSuccess);
 */
export const useUpdateSource = () => {
    const { projectId } = useProjectIdentifier();
    const queryClient = useQueryClient();

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
            ],
            error: {
                notify: true,
            },
        },
    });

    const updateSource = (
        body: { config: SourceConfig; active: boolean; sourceId: string },
        onSuccess?: () => void
    ) => {
        updateSourceMutation.mutate(
            {
                body: {
                    config: body.config,
                    active: body.active,
                },
                params: {
                    path: {
                        project_id: projectId,
                        source_id: body.sourceId,
                    },
                },
            },
            {
                onSuccess: () => {
                    // Invalidate frames cache for the specific source
                    queryClient.invalidateQueries({
                        queryKey: getQueryKey([
                            'get',
                            '/api/v1/projects/{project_id}/sources/{source_id}/frames',
                            {
                                params: {
                                    path: {
                                        project_id: projectId,
                                        source_id: body.sourceId,
                                    },
                                },
                            },
                        ]),
                    });
                    onSuccess?.();
                },
            }
        );
    };

    return {
        mutate: updateSource,
        isPending: updateSourceMutation.isPending,
    };
};
