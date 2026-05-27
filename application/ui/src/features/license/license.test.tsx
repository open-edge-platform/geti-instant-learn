/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@/test-utils';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { LicenseContent } from './license.component';

const renderLicense = (onAccept = vi.fn(), isAccepting = false) => {
    render(<LicenseContent onAccept={onAccept} isAccepting={isAccepting} />);
};

describe('LicenseContent', () => {
    it('renders the acknowledgement text and license links', () => {
        renderLicense();

        expect(screen.getByText(/Intel Simplified Software License/i)).toBeInTheDocument();
    });

    it('calls onAccept when Accept and continue is pressed', async () => {
        const onAccept = vi.fn();
        renderLicense(onAccept);

        await userEvent.click(screen.getByRole('button', { name: /Accept and continue/i }));

        expect(onAccept).toHaveBeenCalledTimes(1);
    });
});
