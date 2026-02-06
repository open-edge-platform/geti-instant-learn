/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';

export const useGetActiveFrame = (sourceId: string) => {
    const { projectId } = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/sources/{source_id}/frames/index', {
        params: {
            path: {
                project_id: projectId,
                source_id: sourceId,
            },
        },
    });
};
