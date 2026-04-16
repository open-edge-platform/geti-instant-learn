/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SourceConfig } from '@/api';
import { useProjectIdentifier } from '@/hooks';

/**
 * Hook for connecting/activating sources dynamically when sourceId isn't known at hook initialization.
 * For editing a specific source, use useUpdateSource(sourceId) instead.
 */
export const useConnectSource = () => {
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
            ],
            error: {
                notify: true,
            },
        },
    });

    const connectSource = (
        sourceId: string,
        body: { config: SourceConfig; active: boolean },
        onSuccess?: () => void
    ) => {
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
        mutate: connectSource,
        isPending: updateSourceMutation.isPending,
    };
};
