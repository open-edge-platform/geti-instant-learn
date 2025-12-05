/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SourceConfig } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

export const useUpdateSource = () => {
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

    const updateSource = (
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
        mutate: updateSource,
        isPending: updateSourceMutation.isPending,
    };
};
