/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ModelStatusType } from '@/api';
import { useProjectIdentifier } from '@/hooks';

/**
 * Fetch the latest model status snapshot for the current project.
 *
 * Used as the hydration value before the SSE stream pushes the first event,
 * and as a fallback when EventSource is unavailable (e.g. during tests or
 * if a proxy strips ``text/event-stream``).
 */
export const useModelStatusSnapshot = (): ModelStatusType | undefined => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useQuery(
        'get',
        '/api/v1/projects/{project_id}/model-status',
        {
            params: { path: { project_id: projectId } },
        },
        {
            // Snapshot is short-lived — refetch on focus to recover after sleep.
            staleTime: 5_000,
            refetchOnWindowFocus: true,
        }
    );

    return data;
};
