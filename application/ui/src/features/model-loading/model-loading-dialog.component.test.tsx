/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { renderHook } from '@/test-utils';
import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../setup-test';
import { useShowModelLoadingDialog } from './model-loading-dialog.component';

describe('useShowModelLoadingDialog', () => {
    beforeEach(() => {
        vi.useFakeTimers({ shouldAdvanceTime: true });
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('returns false when the model is not loading', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/model-status', () => HttpResponse.json({ status: 'ready' }))
        );

        const { result } = renderHook(() => useShowModelLoadingDialog());

        await act(async () => {
            await vi.advanceTimersByTimeAsync(500);
        });

        expect(result.current).toBe(false);
    });

    it('returns true after the spin-delay when the model is loading', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/model-status', () => HttpResponse.json({ status: 'loading' }))
        );

        const { result } = renderHook(() => useShowModelLoadingDialog());

        expect(result.current).toBe(false);

        await act(async () => {
            await vi.advanceTimersByTimeAsync(500);
        });

        await waitFor(() => {
            expect(result.current).toBe(true);
        });
    });
});
