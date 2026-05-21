/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelListType } from '@/api';
import { useProjectIdentifier } from '@/hooks';

export const useGetModels = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/models', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return (data as ModelListType).models;
};
