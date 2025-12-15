/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SourceConfig } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { v4 as uuid } from 'uuid';

export const useCreateSource = () => {
    const { projectId } = useProjectIdentifier();
    const createSourceMutation = $api.useMutation('post', '/api/v1/projects/{project_id}/sources', {
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

    const createSource = (config: SourceConfig, onSuccess?: () => void) => {
        createSourceMutation.mutate(
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
        mutate: createSource,
        isPending: createSourceMutation.isPending,
    };
};
