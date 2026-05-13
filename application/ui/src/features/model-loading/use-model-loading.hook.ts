/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import type { QueryClient } from '@tanstack/react-query';

const POLL_MS = 1_000;
const MAX_ACTIVE_POLL_MS = 60_000;

/**
 * Build the query options for the model-status endpoint.
 * Used to derive the query key and for optimistic cache updates.
 */
const modelStatusOptions = (projectId: string) =>
    $api.queryOptions('get', '/api/v1/projects/{project_id}/model-status', {
        params: { path: { project_id: projectId } },
    });

/**
 * Optimistically set model-status to `{ loading: true }` so the blocking
 * dialog appears immediately. The query's `refetchInterval` will then poll
 * until the backend confirms the actual state.
 *
 * Call this from mutation `onSuccess` callbacks for any operation that
 * triggers a model reload (prompt CRUD, model create/update, etc.).
 */
export const setModelLoading = (queryClient: QueryClient, projectId: string): void => {
    const { queryKey } = modelStatusOptions(projectId);
    queryClient.setQueryData(queryKey, { loading: true });
};

/**
 * Returns `true` while the inference model is being (re)prepared.
 *
 * Polling strategy:
 *   - Idle (`loading` is false): no polling.
 *   - Active loading (`loading` is true): poll every POLL_MS, capped by
 *     MAX_ACTIVE_POLL_MS to avoid runaway polling against a stuck backend.
 */
export const useModelLoading = (): boolean => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useQuery(
        'get',
        '/api/v1/projects/{project_id}/model-status',
        { params: { path: { project_id: projectId } } },
        {
            refetchInterval: (query) => {
                if (!query.state.data?.loading) return false;
                const firstLoadingAt = query.state.dataUpdatedAt;
                if (firstLoadingAt > 0 && Date.now() - firstLoadingAt > MAX_ACTIVE_POLL_MS) {
                    return false;
                }
                return POLL_MS;
            },
            refetchIntervalInBackground: false,
        }
    );

    return data?.loading ?? false;
};
