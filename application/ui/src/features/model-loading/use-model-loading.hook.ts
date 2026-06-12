/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import type { QueryClient } from '@tanstack/react-query';

const POLL_MS = 1_000;

/**
 * Build the query options for the model-status endpoint.
 * Used to derive the query key and for optimistic cache updates.
 */
const modelStatusOptions = (projectId: string) =>
    $api.queryOptions('get', '/api/v1/projects/{project_id}/model-status', {
        params: { path: { project_id: projectId } },
    });

/**
 * Optimistically set model-status to `{ status: 'loading' }` so the blocking
 * dialog appears immediately. The query's `refetchInterval` will then poll
 * until the backend confirms the actual state.
 *
 * Call this from mutation `onSuccess` callbacks for any operation that
 * triggers a model reload (prompt CRUD, model update, etc.).
 */
export const setModelLoading = (queryClient: QueryClient, projectId: string): void => {
    const { queryKey } = modelStatusOptions(projectId);
    queryClient.setQueryData(queryKey, { status: 'loading' });
};

/**
 * Returns `true` while the inference model is being (re)prepared.
 *
 * Polling strategy:
 *   - Idle (`status` is not `loading`): no polling.
 *   - Active loading (`status` is `loading`): poll every POLL_MS.
 */
export const useModelLoading = (): boolean => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useQuery(
        'get',
        '/api/v1/projects/{project_id}/model-status',
        { params: { path: { project_id: projectId } } },
        {
            refetchInterval: (query) => (query.state.data?.status === 'loading' ? POLL_MS : false),
            refetchIntervalInBackground: false,
        }
    );

    return data?.status === 'loading';
};
