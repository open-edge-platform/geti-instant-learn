/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from 'react';

import { baseUrl, ModelStatusType } from '@/api';

const RECONNECT_DELAY_MS = 2_000;

/**
 * Subscribe to the project-scoped model status SSE stream.
 *
 * Returns the latest snapshot pushed by the backend, or ``undefined`` while
 * the connection is being established. Auto-reconnects on transient errors
 * with a fixed backoff; the snapshot REST query in
 * ``useModelStatusSnapshot`` keeps the UI hydrated in the meantime.
 */
export const useModelStatusStream = (projectId: string): ModelStatusType | undefined => {
    const [snapshot, setSnapshot] = useState<ModelStatusType | undefined>(undefined);

    useEffect(() => {
        if (typeof window === 'undefined' || typeof EventSource === 'undefined') {
            return;
        }

        let source: EventSource | null = null;
        let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
        let cancelled = false;

        const connect = () => {
            if (cancelled) return;

            const url = `${baseUrl}/api/v1/projects/${projectId}/model-status/stream`;
            source = new EventSource(url, { withCredentials: false });

            source.onmessage = (event) => {
                try {
                    const parsed = JSON.parse(event.data) as ModelStatusType;
                    setSnapshot(parsed);
                } catch (err) {
                    // Malformed payload — log but keep the stream open.
                    console.warn('Failed to parse model status SSE payload', err);
                }
            };

            source.onerror = () => {
                // EventSource attempts its own reconnects, but on hard errors
                // (4xx/5xx, project deleted) it stays in CLOSED. Force a
                // manual reconnect after a short delay.
                source?.close();
                source = null;
                if (!cancelled) {
                    reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
                }
            };
        };

        connect();

        return () => {
            cancelled = true;
            if (reconnectTimer !== null) {
                clearTimeout(reconnectTimer);
            }
            source?.close();
        };
    }, [projectId]);

    return snapshot;
};
