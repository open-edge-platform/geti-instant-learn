/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelType } from '@/api';
import { setModelLoading } from '@/features/model-loading';
import { useProjectIdentifier } from '@/hooks';
import { useQueryClient } from '@tanstack/react-query';

// TODO: Figure out if the user will ever create a model or if we will provide pre-trained models only
export const useCreateModel = () => {
    const { projectId } = useProjectIdentifier();
    const queryClient = useQueryClient();

    const addModelMutation = $api.useMutation('post', '/api/v1/projects/{project_id}/models', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/models', { params: { path: { project_id: projectId } } }],
            ],
        },
        onSuccess: () => {
            setModelLoading(queryClient, projectId);
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
