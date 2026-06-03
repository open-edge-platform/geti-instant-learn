/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { usePrefetchQuery, useSuspenseQuery } from '@tanstack/react-query';

const sinksQueryOptions = (projectId: string) => {
    return $api.queryOptions('get', '/api/v1/projects/{project_id}/sinks', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });
};

export const usePrefetchSinks = () => {
    const { projectId } = useProjectIdentifier();

    return usePrefetchQuery(sinksQueryOptions(projectId));
};

export const useSinks = () => {
    const { projectId } = useProjectIdentifier();

    return useSuspenseQuery(sinksQueryOptions(projectId));
};
