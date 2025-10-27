/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen, waitForElementToBeRemoved } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { SelectedFrameProvider } from 'src/features/stream/selected-frame-provider.component';
import { describe, expect, it } from 'vitest';

import { http, server } from '../../setup-test';
import { Sidebar } from './sidebar.component';

const renderSidebar = async () => {
    const app = render(
        <SelectedFrameProvider>
            <Sidebar />
        </SelectedFrameProvider>
    );

    if (screen.getByRole('progressbar')) {
        await waitForElementToBeRemoved(screen.getByRole('progressbar'));
    }

    return app;
};

describe('Sidebar', () => {
    it('renders sidebar with prompt tab for active project', async () => {
        await renderSidebar();

        const promptButton = screen.getByRole('button', { name: /toggle prompt tab/i });
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

        const promptButton = screen.getByRole('button', { name: /toggle prompt tab/i });
        expect(promptButton).toBeDisabled();
    });

    it('expands sidebar content when tab is toggled', async () => {
        await renderSidebar();

        const promptButton = screen.getByRole('button', { name: /toggle prompt tab/i });

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

        const promptButton = screen.getByRole('button', { name: /toggle prompt tab/i });
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('collapses sidebar when same tab is clicked again', async () => {
        await renderSidebar();

        const promptButton = screen.getByRole('button', { name: /toggle prompt tab/i });

        expect(promptButton).toHaveAttribute('aria-pressed', 'true');
        expect(await screen.findByRole('heading', { name: /prompt/i })).toBeInTheDocument();

        fireEvent.click(promptButton);
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('resets sidebar when project changes', async () => {
        const { rerender } = await renderSidebar();

        let promptButton = screen.getByRole('button', { name: /toggle prompt tab/i });
        expect(promptButton).toBeEnabled();
        expect(promptButton).toHaveAttribute('aria-pressed', 'true');

        server.use(
            http.get('/api/v1/projects/{project_id}', () => {
                return HttpResponse.json({
                    id: '2',
                    name: 'Different Project',
                    active: false,
                });
            })
        );

        rerender(<Sidebar />);

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));

        promptButton = screen.getByRole('button', { name: /toggle prompt tab/i });
        expect(promptButton).toBeDisabled();
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('applies correct grid layout classes based on expanded state', async () => {
        await renderSidebar();

        const promptButton = screen.getByRole('button', { name: /toggle prompt tab/i });

        const gridContainer = promptButton.closest('[data-expanded]');
        expect(gridContainer).toHaveAttribute('data-expanded', 'true');

        fireEvent.click(promptButton);
        expect(gridContainer).toHaveAttribute('data-expanded', 'false');

        fireEvent.click(promptButton);
        expect(gridContainer).toHaveAttribute('data-expanded', 'true');
    });

    it('renders only one tab (Prompt) with correct icon', async () => {
        await renderSidebar();

        const buttons = screen.getAllByRole('button');

        expect(buttons).toHaveLength(1);
        expect(buttons[0]).toHaveAttribute('aria-label', 'Toggle Prompt tab');
    });
});
