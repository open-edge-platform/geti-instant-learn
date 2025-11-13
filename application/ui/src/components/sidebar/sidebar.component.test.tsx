/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { SelectedFrameProvider } from 'src/features/stream/selected-frame-provider.component';

import { http, server } from '../../setup-test';
import { Sidebar } from './sidebar.component';

const INACTIVE_PROJECT_RESPONSE = {
    id: '1',
    name: 'Inactive Project',
    active: false,
};

const renderSidebar = () => {
    return render(
        <SelectedFrameProvider>
            <Sidebar />
        </SelectedFrameProvider>
    );
};

describe('Sidebar', () => {
    it('renders sidebar with prompt tab for active project', async () => {
        renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /prompt tab/i });
        expect(promptButton).toBeInTheDocument();
        expect(promptButton).toBeEnabled();
    });

    it('disables prompt tab when project is inactive', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}', () => {
                return HttpResponse.json(INACTIVE_PROJECT_RESPONSE);
            })
        );

        renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /prompt tab/i });
        expect(promptButton).toBeDisabled();
    });

    it('shows sidebar content when tab is selected', async () => {
        renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /prompt tab/i });

        expect(promptButton).toHaveAttribute('aria-pressed', 'true');
        expect(await screen.findByRole('heading', { name: /prompt/i })).toBeInTheDocument();
    });

    it('does not show sidebar content for inactive project', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}', () => {
                return HttpResponse.json(INACTIVE_PROJECT_RESPONSE);
            })
        );

        renderSidebar();

        expect(screen.queryByRole('heading', { name: /prompt/i })).not.toBeInTheDocument();

        const promptButton = await screen.findByRole('button', { name: /prompt tab/i });
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');
    });
});
