/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, renderHook } from '@/test-utils';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http, server } from '../../../setup-test';
import { ModelLoadingDialog, useShowModelLoadingDialog } from './model-loading-dialog.component';

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

describe('ModelLoadingDialog', () => {
    it('shows error dialog with correct heading, message and retry button when model status is error', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/model-status', () =>
                HttpResponse.json({ status: 'error', error_message: 'Failed to load model weights' })
            )
        );

        render(<ModelLoadingDialog />);

        expect(await screen.findByRole('dialog', { name: 'Model loading error' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Model loading error' })).toBeInTheDocument();
        expect(screen.getByText('Failed to load model weights')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    });

    it('calls reload endpoint with correct project id on retry and hides error dialog when model loads successfully', async () => {
        let statusRequestCount = 0;
        let capturedProjectId: string | undefined;

        server.use(
            http.get('/api/v1/projects/{project_id}/model-status', () => {
                statusRequestCount++;

                if (statusRequestCount <= 1) {
                    return HttpResponse.json({ status: 'error', error_message: 'Something went wrong' });
                }

                return HttpResponse.json({ status: 'ready' });
            }),
            http.post('/api/v1/projects/{project_id}/reload', ({ params }) => {
                capturedProjectId = params.project_id;

                return HttpResponse.json({}, { status: 202 });
            })
        );

        render(<ModelLoadingDialog />);

        await userEvent.click(await screen.findByRole('button', { name: 'Retry' }));

        await waitFor(() => {
            expect(capturedProjectId).toBe('1');
        });

        await waitFor(() => {
            expect(screen.queryByRole('dialog', { name: 'Model loading error' })).not.toBeInTheDocument();
        });
    });
});
