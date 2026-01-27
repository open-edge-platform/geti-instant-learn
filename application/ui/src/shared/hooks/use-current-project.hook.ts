/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';

export const useCurrentProject = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });
};
