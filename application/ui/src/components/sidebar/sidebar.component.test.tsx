/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen } from '@testing-library/react';
import { SelectedFrameProvider } from 'src/shared/selected-frame-provider.component';

import { Sidebar } from './sidebar.component';

const renderSidebar = () => {
    return render(
        <SelectedFrameProvider>
            <Sidebar />
        </SelectedFrameProvider>
    );
};

describe('Sidebar', () => {
    it('renders sidebar with prompt tab', async () => {
        renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });
        expect(promptButton).toBeInTheDocument();
        expect(promptButton).toBeEnabled();
    });

    it('expands sidebar content when tab is toggled', async () => {
        renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });

        expect(promptButton).toHaveAttribute('aria-pressed', 'true');
        expect(await screen.findByRole('heading', { name: /prompt/i })).toBeInTheDocument();

        fireEvent.click(promptButton);
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');

        fireEvent.click(promptButton);
        expect(promptButton).toHaveAttribute('aria-pressed', 'true');
        expect(await screen.findByRole('heading', { name: /prompt/i })).toBeInTheDocument();
    });

    it('collapses sidebar when same tab is clicked again', async () => {
        renderSidebar();

        const promptButton = await screen.findByRole('button', { name: /toggle prompt tab/i });

        expect(promptButton).toHaveAttribute('aria-pressed', 'true');
        expect(await screen.findByRole('heading', { name: /prompt/i })).toBeInTheDocument();

        fireEvent.click(promptButton);
        expect(promptButton).toHaveAttribute('aria-pressed', 'false');
    });
});
