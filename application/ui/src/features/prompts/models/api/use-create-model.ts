/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelType } from '@/api';
import { useProjectIdentifier } from '@/hooks';

// TODO: Figure out if the user will ever create a model or if we will provide pre-trained models only
export const useCreateModel = () => {
    const { projectId } = useProjectIdentifier();
    const addModelMutation = $api.useMutation('post', '/api/v1/projects/{project_id}/models', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/models', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    return (model: ModelType) =>
        addModelMutation.mutate({
            body: model,
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
};
