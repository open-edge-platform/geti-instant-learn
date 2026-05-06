/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@/test-utils';
import { screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../setup-test';
import { ModelStatusBanner } from './model-status-banner.component';
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
    it('shows the ready label and keeps it visible', async () => {
        mockStatus({
            state: 'ready',
            message: 'Model matcher ready on cuda',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Ready');
        expect(status).toHaveTextContent(/Ready/i);
    });

    it('shows a loading label and a spinner while the model is loading', async () => {
        mockStatus({
            state: 'loading_model',
            message: 'Loading model matcher on cuda…',
            model_name: 'matcher',
            device: 'cuda',
            project_id: '1',
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Loading…');
        // Spinner is present while busy.
        expect(screen.getByLabelText('Loading')).toBeVisible();
    });

    it('shows the build-prompts label while the reference batch is being built', async () => {
        mockStatus({
            state: 'loading_reference_batch',
            message: 'Building reference batch from prompts…',
            project_id: '1',
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Building prompts…');
    });

    it('shows the error label on ERROR state', async () => {
        mockStatus({
            state: 'error',
            message: 'Model failed to load: boom',
            project_id: '1',
            error: { code: 'RuntimeError', detail: 'boom' },
        });

        renderBanner();

        const status = await screen.findByRole('status');
        expect(status).toHaveAttribute('aria-label', 'Model Error');
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
