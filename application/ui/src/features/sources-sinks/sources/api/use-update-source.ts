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
 * Can be used in two ways:
 * 1. With sourceId at hook creation: `useUpdateSource(sourceId)` - for editing specific sources
 * 2. Without sourceId: `useUpdateSource()` - for dynamically connecting sources, pass sourceId to mutate()
 */
export const useUpdateSource = (sourceId?: string) => {
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
        bodyOrSourceId: { config: SourceConfig; active: boolean } | string,
        bodyOrOnSuccess?: { config: SourceConfig; active: boolean } | (() => void),
        onSuccessParam?: () => void
    ) => {
        // Handle overloaded parameters
        let finalSourceId: string;
        let finalBody: { config: SourceConfig; active: boolean };
        let finalOnSuccess: (() => void) | undefined;

        if (typeof bodyOrSourceId === 'string') {
            // Called as: mutate(sourceId, body, onSuccess)
            finalSourceId = bodyOrSourceId;
            finalBody = bodyOrOnSuccess as { config: SourceConfig; active: boolean };
            finalOnSuccess = onSuccessParam;
        } else {
            // Called as: mutate(body, onSuccess)
            if (!sourceId) {
                throw new Error('sourceId must be provided either at hook creation or mutation time');
            }
            finalSourceId = sourceId;
            finalBody = bodyOrSourceId;
            finalOnSuccess = bodyOrOnSuccess as (() => void) | undefined;
        }

        updateSourceMutation.mutate(
            {
                body: finalBody,
                params: {
                    path: {
                        project_id: projectId,
                        source_id: finalSourceId,
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
                                        source_id: finalSourceId,
                                    },
                                },
                            },
                        ]),
                    });
                    finalOnSuccess?.();
                },
            }
        );
    };

    return {
        mutate: updateSource,
        isPending: updateSourceMutation.isPending,
    };
};
