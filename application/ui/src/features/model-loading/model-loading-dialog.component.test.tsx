/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@/test-utils';
import { screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../setup-test';
import { ModelLoadingDialog } from './model-loading-dialog.component';

describe('ModelLoadingDialog', () => {
    beforeEach(() => {
        vi.useFakeTimers({ shouldAdvanceTime: true });
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('does not render the dialog when the model is not loading', async () => {
        server.use(http.get('/api/v1/projects/{project_id}/model-status', () => HttpResponse.json({ loading: false })));

        render(<ModelLoadingDialog />);

        // Spin-delay's entry delay (300ms) elapses; nothing should appear.
        await vi.advanceTimersByTimeAsync(400);

        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('renders the blocking dialog after the spin-delay when the model is loading', async () => {
        server.use(http.get('/api/v1/projects/{project_id}/model-status', () => HttpResponse.json({ loading: true })));

        render(<ModelLoadingDialog />);

        // Before the 300ms spin-delay elapses, dialog must not be visible.
        await vi.advanceTimersByTimeAsync(100);
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

        // After the spin-delay, the dialog appears.
        await vi.advanceTimersByTimeAsync(400);
        await waitFor(() => {
            expect(screen.getByRole('dialog')).toBeVisible();
        });
        expect(screen.getByText(/Loading model/i)).toBeVisible();
        expect(screen.getByText(/Please wait/i)).toBeVisible();
    });
});
