/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SourceType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { usePrefetchQuery } from '@tanstack/react-query';

export const useAvailableSources = (sourceType: SourceType) => {
    return $api.useSuspenseQuery('get', '/api/v1/system/source-types/{source_type}/sources', {
        params: {
            path: {
                source_type: sourceType,
            },
        },
    });
};

export const usePrefetchAvailableSources = (sourceType: SourceType) => {
    return $api.useQuery('get', '/api/v1/system/source-types/{source_type}/sources', {
        params: {
            path: {
                source_type: sourceType,
            },
        },
    });
};

export const usePrefetchSources = () => {
    const { projectId } = useProjectIdentifier();

    return usePrefetchQuery(
        $api.queryOptions('get', '/api/v1/projects/{project_id}/sources', {
            params: {
                path: {
                    project_id: projectId,
                },
            },
        })
    );
};
