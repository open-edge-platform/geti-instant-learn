/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, type paths } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import type { QueryKey } from '@/query-client/query-client.interface';
import type { QueryClient } from '@tanstack/react-query';

export const MODEL_STATUS_PATH = '/api/v1/projects/{project_id}/model-status' as const;

/**
 * Build the query key for the model-status endpoint so mutation hooks
 * can include it in their `meta.invalidates` array.
 *
 * The QueryClient watches for this key in `meta.invalidates` and automatically
 * starts a probe window — no extra per-hook code required.
 */
export const modelStatusQueryKey = (projectId: string): QueryKey<paths> => [
    'get',
    MODEL_STATUS_PATH,
    { params: { path: { project_id: projectId } } },
];

const POLL_MS = 1_000;
const PROBE_INTERVAL_MS = 1_000;
const PROBE_ATTEMPTS = 5;
const MAX_ACTIVE_POLL_MS = 60_000;

let probeTimer: ReturnType<typeof setInterval> | null = null;
let probeAttemptsLeft = 0;

/**
 * Open a probe window: actively refetch the model-status query several times
 * to catch the backend's asynchronous transition into `loading: true`.
 *
 * Encapsulates the (necessarily module-scoped) timer so callers — and tests —
 * have a single entry/exit point. Once `loading: true` is observed the query's
 * own `refetchInterval` takes over and the probe becomes irrelevant.
 */
export const startModelStatusProbe = (queryClient: QueryClient): void => {
    probeAttemptsLeft = PROBE_ATTEMPTS;
    if (probeTimer !== null) {
        return;
    }
    probeTimer = setInterval(() => {
        if (probeAttemptsLeft <= 0) {
            stopModelStatusProbe();
            return;
        }
        probeAttemptsLeft -= 1;
        void queryClient.refetchQueries({ queryKey: ['get', MODEL_STATUS_PATH] });
    }, PROBE_INTERVAL_MS);
};

/** Exposed for tests / hot-reload cleanup. */
export const stopModelStatusProbe = (): void => {
    if (probeTimer !== null) {
        clearInterval(probeTimer);
        probeTimer = null;
    }
    probeAttemptsLeft = 0;
};

/**
 * Returns `true` while the inference model is being (re)prepared.
 *
 * Polling strategy:
 *   - Idle: no polling.
 *   - Probe window (opened by the QueryClient when a mutation invalidates the
 *     model-status key): up to PROBE_ATTEMPTS active refetches at
 *     PROBE_INTERVAL_MS intervals.
 *   - Active loading (`data.loading === true`): poll every POLL_MS, capped by
 *     MAX_ACTIVE_POLL_MS to avoid runaway polling against a stuck backend.
 */
export const useModelLoading = (): boolean => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useQuery(
        'get',
        MODEL_STATUS_PATH,
        {
            params: { path: { project_id: projectId } },
        },
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
