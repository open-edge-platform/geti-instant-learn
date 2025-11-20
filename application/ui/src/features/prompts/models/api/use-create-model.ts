/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { v4 as uuid } from 'uuid';

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

    const mockModel: ModelType = {
        id: uuid(),
        config: {
            mask_similarity_threshold: 0.38,
            model_type: 'matcher',
            num_background_points: 2,
            num_foreground_points: 40,
            precision: 'bf16',
        },
        active: true,
        name: `My model ${uuid().slice(0, 8)}`,
    };

    return (model?: ModelType) =>
        addModelMutation.mutate({
            body: model ?? mockModel,
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
};
