/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, LabelType } from '@/api';
import { useProjectIdentifier } from '@/hooks';

export const useProjectLabels = (): LabelType[] => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/labels', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data.labels ?? [];
};
