/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { getMockedMatcherModel, getMockedModel, getMockedSam3Model, render } from '@/test-utils';
import { fireEvent, screen, within } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../../../setup-test';
import { ModelToolbar } from './model-toolbar.component';

const mockProjectWithMode = (promptMode: 'TEXT' | 'VISUAL') => {
    server.use(
        http.get('/api/v1/projects/{project_id}', () => {
            return HttpResponse.json({
                id: '1',
                name: 'Project #1',
                active: true,
                device: 'cpu',
                prompt_mode: promptMode,
            });
        })
    );
};
const renderToolbar = () => render(<ModelToolbar />, { route: '/projects/1', path: '/projects/:projectId' });
describe('ModelToolbar', () => {
    it('does not render picker if there are no models', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/models', () => {
                return HttpResponse.json({ models: [], pagination: { total: 0, count: 0, offset: 0, limit: 10 } });
            })
        );
        renderToolbar();
        expect(await screen.findByText(/No models available/i)).toBeVisible();
    });
    it('renders models correctly', async () => {
        renderToolbar();
        const pickerButton = await screen.findByRole('button', { name: /Mega model/i });
        fireEvent.click(pickerButton);
        const listbox = screen.getByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Mega model' })).toBeVisible();
    });
    it('changes selected model when a different option is clicked', async () => {
        let activeModelId = 'model-1';
        server.use(
            http.get('/api/v1/projects/{project_id}/models', () => {
                return HttpResponse.json({
                    models: [
                        getMockedModel({ id: 'model-1', name: 'Mega model', active: activeModelId === 'model-1' }),
                        getMockedModel({ id: 'model-2', name: 'Tiny model', active: activeModelId === 'model-2' }),
                    ],
                    pagination: { total: 2, count: 2, offset: 0, limit: 10 },
                });
            }),
            http.put('/api/v1/projects/{project_id}/models/{model_id}', async ({ request, params }) => {
                activeModelId = params.model_id as string;
                const body = await request.json();
                return HttpResponse.json(body);
            })
        );
        renderToolbar();
        const pickerButton = await screen.findByRole('button', { name: /Mega model/i });
        fireEvent.click(pickerButton);
        const listbox = screen.getByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Mega model' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'Tiny model' })).toBeVisible();
        fireEvent.click(screen.getByRole('option', { name: 'Tiny model' }));
        const pickerButtonTwo = await screen.findByRole('button', { name: /Tiny model/i });
        expect(within(pickerButtonTwo).getByText('Tiny model')).toBeVisible();
    });
    it('shows only models returned by the API (backend filters by prompt_mode)', async () => {
        mockProjectWithMode('VISUAL');
        // Backend already filters — only visual models are returned for VISUAL mode
        server.use(
            http.get('/api/v1/projects/{project_id}/models', () => {
                return HttpResponse.json({
                    models: [
                        getMockedMatcherModel({ id: 'matcher-1', name: 'Matcher', active: true }),
                        getMockedSam3Model({ id: 'sam3-1', name: 'SAM3', active: false }),
                    ],
                    pagination: { total: 2, count: 2, offset: 0, limit: 10 },
                });
            })
        );
        renderToolbar();
        const pickerButton = await screen.findByRole('button', { name: /Matcher/i });
        fireEvent.click(pickerButton);
        const listbox = screen.getByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Matcher' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'SAM3' })).toBeVisible();
    });
    it('shows the backend-determined active model after mode switch', async () => {
        mockProjectWithMode('TEXT');
        // Backend returns only SAM3 for TEXT mode with active=true (backend already switched)
        server.use(
            http.get('/api/v1/projects/{project_id}/models', () => {
                return HttpResponse.json({
                    models: [getMockedSam3Model({ id: 'sam3-1', name: 'SAM3', active: true })],
                    pagination: { total: 1, count: 1, offset: 0, limit: 10 },
                });
            })
        );
        renderToolbar();
        expect(await screen.findByRole('button', { name: /SAM3/i })).toBeVisible();
    });
});
