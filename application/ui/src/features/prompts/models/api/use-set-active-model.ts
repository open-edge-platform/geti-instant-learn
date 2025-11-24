/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

const useSetActiveModelQuery = () => {
    return $api.useMutation('put', '/api/v1/projects/{project_id}/models/{model_id}');
};

export const useSetActiveModel = () => {
    const { projectId } = useProjectIdentifier();
    const updateModelMutation = useSetActiveModelQuery();

    return (model: ModelType) => {
        const { id, ...rest } = model;

        // We need send everything BUT the `id`
        return updateModelMutation.mutate({
            body: { ...rest, active: true },
            params: {
                path: {
                    project_id: projectId,
                    model_id: model.id,
                },
            },
        });
    };
};
