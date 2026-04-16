/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { getMockedModel, getMockedSam3Model, render } from '@/test-utils';
import type { MatcherModel } from '@/api';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../../../setup-test';
import { ModelToolbar } from './model-toolbar.component';

const renderToolbar = (route = '/projects/1?mode=visual') =>
    render(<ModelToolbar />, { route, path: '/projects/:projectId' });

describe('ModelToolbar', () => {
    it('does not render picker if there are no models', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/models', () => {
                return HttpResponse.json({ models: [], pagination: { total: 0, count: 0, offset: 0, limit: 10 } });
            })
        );

        render(<ModelToolbar />);

        expect(await screen.findByText(/No models available/i)).toBeVisible();
    });

    it('renders models correctly', async () => {
        renderToolbar();

        const pickerButton = await screen.findByRole('button', { name: /Mega model/i }, { timeout: 5000 });
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

    it('only shows visual-compatible models in visual prompt mode', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/models', () => {
                return HttpResponse.json({
                    models: [
                        getMockedModel({
                            id: 'matcher-1',
                            name: 'Matcher',
                            active: true,
                            config: { ...getMockedModel().config, model_type: 'matcher' } as MatcherModel['config'],
                        }),
                        getMockedSam3Model({ id: 'sam3-1', name: 'SAM3', active: false }),
                    ],
                    pagination: { total: 2, count: 2, offset: 0, limit: 10 },
                });
            })
        );

        // Default mode is "visual" — matcher supports visual_polygon, SAM3 supports visual_rectangle
        // Both should be visible
        renderToolbar('/projects/1?mode=visual');

        const pickerButton = await screen.findByRole('button', { name: /Matcher/i });
        fireEvent.click(pickerButton);

        const listbox = screen.getByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Matcher' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'SAM3' })).toBeVisible();
    });

    it('only shows text-compatible models in text prompt mode', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/models', () => {
                return HttpResponse.json({
                    models: [
                        getMockedModel({
                            id: 'matcher-1',
                            name: 'Matcher',
                            active: true,
                            config: { ...getMockedModel().config, model_type: 'matcher' } as MatcherModel['config'],
                        }),
                        getMockedSam3Model({ id: 'sam3-1', name: 'SAM3', active: false }),
                    ],
                    pagination: { total: 2, count: 2, offset: 0, limit: 10 },
                });
            })
        );

        // Text mode — only SAM3 supports "text"
        renderToolbar('/projects/1?mode=text');

        // SAM3 is the only text-compatible model
        await waitFor(() => {
            expect(screen.queryByText('Matcher')).not.toBeInTheDocument();
        });

        expect(await screen.findByRole('button', { name: /SAM3/i })).toBeVisible();
    });
});
