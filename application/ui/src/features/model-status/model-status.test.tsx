/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@/test-utils';
import { act, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../setup-test';
import { ModelStatusBanner } from './model-status-banner.component';
import { ModelStatusBlockingOverlay } from './model-status-blocking-overlay.component';
import { ModelStatusProvider } from './model-status-provider.component';

const mockStatus = (body: Record<string, unknown>) =>
    server.use(http.get('/api/v1/projects/{project_id}/model-status', () => HttpResponse.json(body)));

const renderBanner = () =>
    render(
        <ModelStatusProvider>
            <ModelStatusBanner />
        </ModelStatusProvider>
    );

describe('ModelStatusBanner', () => {
    it('shows the ready label with model and device and keeps it visible', async () => {
        mockStatus({
            state: 'ready',
            message: 'Model matcher ready on cuda',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Ready · matcher on cuda');
        expect(status).toHaveTextContent(/Ready · matcher on cuda/i);
    });

    it('falls back to the bare state label when model and device are unknown', async () => {
        mockStatus({
            state: 'ready',
            message: 'Model ready',
            project_id: '1',
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Ready');
    });

    it('shows a loading label with model/device and a spinner while the model is loading', async () => {
        mockStatus({
            state: 'loading_model',
            message: 'Loading model matcher on cuda…',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Loading matcher on cuda…');
        expect(screen.getByLabelText('Loading')).toBeVisible();
    });

    it('shows the build-prompts label while the reference batch is being built', async () => {
        mockStatus({
            state: 'loading_reference_batch',
            message: 'Building reference batch from prompts…',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Building prompts · matcher on cuda');
    });

    it('shows the error label with model/device on ERROR state', async () => {
        mockStatus({
            state: 'error',
            message: 'Model failed to load: boom',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
            error: { code: 'RuntimeError', detail: 'boom' },
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Error · matcher on cuda');
    });

    it('shows the idle label when no model is loaded', async () => {
        mockStatus({
            state: 'idle',
            message: 'No active model',
            project_id: '1',
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Idle');
    });

    it('renders nothing while the initial snapshot is still loading', async () => {
        // Long-pending response: the snapshot query never resolves before assertion.
        server.use(
            http.get('/api/v1/projects/{project_id}/model-status', async () => {
                await new Promise((resolve) => setTimeout(resolve, 1000));
                return HttpResponse.json({ state: 'idle', message: 'No active model', project_id: '1' });
            })
        );

        renderBanner();

        // Give React a tick to settle the loading state.
        await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument());
    });
});

const renderOverlay = () =>
    render(
        <ModelStatusProvider>
            <ModelStatusBlockingOverlay />
        </ModelStatusProvider>
    );

describe('ModelStatusBlockingOverlay', () => {
    beforeEach(() => {
        vi.useFakeTimers({ shouldAdvanceTime: true });
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('appears on loading_model after debounce with role="alertdialog"', async () => {
        mockStatus({
            state: 'loading_model',
            message: 'Loading model matcher on cuda…',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
        });

        renderOverlay();

        // Wait for REST snapshot to arrive, then advance past the 200ms debounce.
        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
        act(() => vi.advanceTimersByTime(250));

        await waitFor(() => {
            const dialog = screen.getByRole('alertdialog');
            expect(dialog).toBeVisible();
            expect(dialog).toHaveAttribute('aria-modal', 'true');
        });
    });

    it('appears on loading_reference_batch and shows model + device', async () => {
        mockStatus({
            state: 'loading_reference_batch',
            message: 'Building reference batch…',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
        });

        renderOverlay();

        act(() => vi.advanceTimersByTime(250));

        await waitFor(() => {
            const dialog = screen.getByRole('alertdialog');
            expect(dialog).toBeVisible();
            expect(dialog).toHaveTextContent(/Matcher \(CUDA\)/);
            expect(dialog).toHaveTextContent(/Preparing.*for inference/);
        });
    });

    it('does not appear on ready state', async () => {
        mockStatus({
            state: 'ready',
            message: 'Model ready',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
        });

        renderOverlay();

        act(() => vi.advanceTimersByTime(500));

        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
    });

    it('does not appear on idle state', async () => {
        mockStatus({
            state: 'idle',
            message: 'No active model',
            project_id: '1',
        });

        renderOverlay();

        act(() => vi.advanceTimersByTime(500));

        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
    });

    it('does not appear on error state', async () => {
        mockStatus({
            state: 'error',
            message: 'Model failed to load',
            project_id: '1',
        });

        renderOverlay();

        act(() => vi.advanceTimersByTime(500));

        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
    });
});
