/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen, within } from '@testing-library/react';

import { ModelToolbar } from './model-toolbar.component';

describe('ModelToolbar', () => {
    it('renders model picker with default selection', () => {
        render(<ModelToolbar />);

        const pickerButton = screen.getByRole('button', { name: /Model/i });
        expect(pickerButton).toBeVisible();

        expect(within(pickerButton).getByText('DINO v2')).toBeVisible();
        expect(screen.getByText(/Deployed:/i)).toBeVisible();
    });

    it('changes selected model when a different option is clicked', async () => {
        render(<ModelToolbar />);

        const pickerButton = screen.getByRole('button', { name: /Model/i });
        fireEvent.click(pickerButton);

        const listbox = screen.getByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'DINO v2' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'DINO v3' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'DINO v4' })).toBeVisible();

        fireEvent.click(screen.getByRole('option', { name: 'DINO v3' }));

        const pickerButtonTwo = screen.getByRole('button', { name: /Model/i });
        expect(within(pickerButtonTwo).getByText('DINO v3')).toBeVisible();
    });
});
