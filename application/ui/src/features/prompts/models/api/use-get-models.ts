/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelListType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

const useGetModelsQuery = (): ModelListType => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/models', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data;
};

export const useGetModels = () => {
    const { models } = useGetModelsQuery();

    return models;
};
