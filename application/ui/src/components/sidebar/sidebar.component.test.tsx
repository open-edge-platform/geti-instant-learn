/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { SelectedFrameProvider } from 'src/features/stream/selected-frame-provider.component';
import { describe, expect, it } from 'vitest';

import { http, server } from '../../setup-test';
import { Sidebar } from './sidebar.component';

const renderSidebar = async () => {
    return render(
        <SelectedFrameProvider>
            <Sidebar />
        </SelectedFrameProvider>
    );
};

describe('Sidebar', () => {
    it('renders sidebar with prompt tab for active project', async () => {
        await renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });
        expect(promptButton).toBeInTheDocument();
        expect(promptButton).toBeEnabled();
    });

    it('disables prompt tab when project is inactive', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}', () => {
                return HttpResponse.json({
                    id: '1',
                    name: 'Inactive Project',
                    active: false,
                });
            })
        );

        await renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });
        expect(promptButton).toBeDisabled();
    });

    it('expands sidebar content when tab is toggled', async () => {
        await renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });

        expect(promptButton).toHaveAttribute('aria-pressed', 'true');
        expect(await screen.findByRole('heading', { name: /prompt/i })).toBeInTheDocument();

        fireEvent.click(promptButton);
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');

        fireEvent.click(promptButton);
        expect(promptButton).toHaveAttribute('aria-pressed', 'true');
        expect(await screen.findByRole('heading', { name: /prompt/i })).toBeInTheDocument();
    });

    it('does not expand sidebar for inactive project', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}', () => {
                return HttpResponse.json({
                    id: '1',
                    name: 'Inactive Project',
                    active: false,
                });
            })
        );

        await renderSidebar();

        expect(screen.queryByRole('heading', { name: /prompt/i })).not.toBeInTheDocument();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('collapses sidebar when same tab is clicked again', async () => {
        await renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });

        expect(promptButton).toHaveAttribute('aria-pressed', 'true');
        expect(await screen.findByRole('heading', { name: /prompt/i })).toBeInTheDocument();

        fireEvent.click(promptButton);
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('applies correct grid layout classes based on expanded state', async () => {
        await renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });

        const gridContainer = promptButton.closest('[data-expanded]');
        expect(gridContainer).toHaveAttribute('data-expanded', 'true');

        fireEvent.click(promptButton);
        expect(gridContainer).toHaveAttribute('data-expanded', 'false');

        fireEvent.click(promptButton);
        expect(gridContainer).toHaveAttribute('data-expanded', 'true');
    });
});
