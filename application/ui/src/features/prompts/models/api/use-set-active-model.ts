/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { omit } from 'lodash-es';

const useSetActiveModelQuery = () => {
    return $api.useMutation('put', '/api/v1/projects/{project_id}/models/{model_id}');
};

export const useSetActiveModel = () => {
    const { projectId } = useProjectIdentifier();
    const updateModelMutation = useSetActiveModelQuery();

    return (model: ModelType) =>
        updateModelMutation.mutate({
            body: { ...omit(model, ['id']), active: true },
            params: {
                path: {
                    project_id: projectId,
                    model_id: model.id,
                },
            },
        });
};
