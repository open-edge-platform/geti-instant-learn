/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext, useEffect, useMemo, useRef } from 'react';

import { ModelState, ModelStatusType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { toast } from '@geti/ui';

import { useModelStatusSnapshot } from './api/use-model-status-snapshot.hook';
import { useModelStatusStream } from './use-model-status-stream.hook';

export interface ModelStatusContextValue {
    /** Latest status snapshot (from SSE if available, otherwise the REST snapshot). */
    status: ModelStatusType | undefined;
    /** True while the model is loading (controls should be disabled). */
    isBusy: boolean;
    /** True when the model is ready to serve predictions. */
    isReady: boolean;
    /** True when the last transition was an error. */
    isError: boolean;
    /** True when no model is loaded (e.g. passthrough). */
    isIdle: boolean;
}

const BUSY_STATES: readonly ModelState[] = ['loading_reference_batch', 'loading_model'];

const ModelStatusContext = createContext<ModelStatusContextValue | undefined>(undefined);

const computeFlags = (status: ModelStatusType | undefined): Omit<ModelStatusContextValue, 'status'> => {
    const state = status?.state;

    return {
        isBusy: state !== undefined && BUSY_STATES.includes(state),
        isReady: state === 'ready',
        isError: state === 'error',
        isIdle: state === 'idle',
    };
};

/**
 * Provides the live model status to descendants.
 *
 * Combines an initial REST snapshot (for hydration) with a long-lived SSE
 * stream of state transitions. Errors surface as one-shot toasts so users
 * are notified even if they aren't looking at the prompt panel.
 */
export const ModelStatusProvider = ({ children }: { children: ReactNode }) => {
    const { projectId } = useProjectIdentifier();
    const initialSnapshot = useModelStatusSnapshot();
    const liveSnapshot = useModelStatusStream(projectId);

    const status = liveSnapshot ?? initialSnapshot;
    const flags = computeFlags(status);

    // Surface ERROR transitions as a toast (deduped on updated_at + message).
    const lastErrorKey = useRef<string | null>(null);
    useEffect(() => {
        if (status?.state !== 'error') {
            lastErrorKey.current = null;
            return;
        }
        const key = `${status.updated_at ?? ''}|${status.message}`;
        if (lastErrorKey.current === key) {
            return;
        }
        lastErrorKey.current = key;
        toast({ type: 'error', message: status.message });
    }, [status?.state, status?.updated_at, status?.message]);

    const value = useMemo<ModelStatusContextValue>(
        () => ({ status, ...flags }),
        // ``flags`` is recomputed every render; depend on its underlying fields.
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [status, flags.isBusy, flags.isReady, flags.isError, flags.isIdle]
    );

    return <ModelStatusContext.Provider value={value}>{children}</ModelStatusContext.Provider>;
};

export const useModelStatus = (): ModelStatusContextValue => {
    const ctx = useContext(ModelStatusContext);
    if (ctx === undefined) {
        throw new Error('useModelStatus must be used inside ModelStatusProvider');
    }
    return ctx;
};
