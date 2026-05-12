/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';

/**
 * Poll the backend for the model busy-state.
 *
 * Returns `true` while the inference model is being (re)prepared (reference
 * batch building + weight download + model initialization).
 *
 * Polling is kept coarse (1.5s) — the spin-delay applied in the consumer
 * hides the polling latency for short loads. Polling pauses while the tab is
 * not visible to avoid wasted CPU on long sessions.
 */
export const useModelLoading = (): boolean => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useQuery(
        'get',
        '/api/v1/projects/{project_id}/model-status',
        {
            params: { path: { project_id: projectId } },
        },
        {
            refetchInterval: 1500,
            refetchIntervalInBackground: false,
        }
    );

    return data?.loading ?? false;
};
