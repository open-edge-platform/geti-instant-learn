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

describe('ModelStatusBanner', () => {
    it('renders nothing while the model is ready', async () => {
        mockStatus({
            state: 'ready',
            message: 'Model sam3 ready on cuda',
            model_name: 'sam3',
            device: 'cuda',
            project_id: '1',
        });

        render(
            <ModelStatusProvider>
                <ModelStatusBanner />
            </ModelStatusProvider>
        );

        await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument());
        expect(screen.queryByLabelText('Model error')).not.toBeInTheDocument();
    });

    it('shows a loading message while the model is loading', async () => {
        mockStatus({
            state: 'loading_model',
            message: 'Loading model sam3 on cuda…',
            model_name: 'sam3',
            device: 'cuda',
            project_id: '1',
        });

        render(
            <ModelStatusProvider>
                <ModelStatusBanner />
            </ModelStatusProvider>
        );

        expect(await screen.findByText(/Loading model sam3 on cuda/i)).toBeVisible();
        expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'Model loading');
    });

    it('shows a message while the reference batch is being built', async () => {
        mockStatus({
            state: 'loading_reference_batch',
            message: 'Building reference batch from prompts…',
            project_id: '1',
        });

        render(
            <ModelStatusProvider>
                <ModelStatusBanner />
            </ModelStatusProvider>
        );

        expect(await screen.findByText(/Building reference batch/i)).toBeVisible();
    });

    it('shows an error alert on ERROR state', async () => {
        mockStatus({
            state: 'error',
            message: 'Model failed to load: boom',
            project_id: '1',
            error: { code: 'RuntimeError', detail: 'boom' },
        });

        render(
            <ModelStatusProvider>
                <ModelStatusBanner />
            </ModelStatusProvider>
        );

        expect(await screen.findByText(/Model failed to load: boom/i)).toBeVisible();
    });
});
