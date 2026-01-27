/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { getMockedModel, render } from '@/test-utils';
import { fireEvent, screen, within } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../../../setup-test';
import { ModelToolbar } from './model-toolbar.component';

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
        render(<ModelToolbar />);

        const pickerButton = await screen.findByRole('button', { name: /Mega model/i });
        fireEvent.click(pickerButton);

        const listbox = screen.getByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Mega model' })).toBeVisible();
    });

    it('changes selected model when a different option is clicked', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/models', () => {
                return HttpResponse.json({
                    models: [
                        getMockedModel({ id: 'model-1', name: 'Mega model', active: true }),
                        getMockedModel({ id: 'model-2', name: 'Tiny model', active: false }),
                    ],
                    pagination: { total: 0, count: 0, offset: 0, limit: 10 },
                });
            })
        );

        render(<ModelToolbar />);

        const pickerButton = await screen.findByRole('button', { name: /Mega model/i });
        fireEvent.click(pickerButton);

        const listbox = screen.getByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Mega model' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'Tiny model' })).toBeVisible();

        fireEvent.click(screen.getByRole('option', { name: 'Tiny model' }));

        const pickerButtonTwo = screen.getByRole('button', { name: /Tiny model/i });
        expect(within(pickerButtonTwo).getByText('Tiny model')).toBeVisible();
    });
});
