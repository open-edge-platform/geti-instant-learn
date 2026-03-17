/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@/test-utils';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { License } from './license.component';

const renderLicense = (onAccept = vi.fn(), isAccepting = false) => {
    render(<License onAccept={onAccept} isAccepting={isAccepting} />);
};

describe('License', () => {
    it('renders the acknowledgement text and license links', () => {
        renderLicense();

        expect(screen.getByText(/By installing, using, or distributing this application/i)).toBeInTheDocument();
        expect(screen.getByRole('link', { name: /SAM3 License/i })).toHaveAttribute(
            'href',
            'https://github.com/facebookresearch/sam3/blob/main/LICENSE'
        );
        expect(screen.getByRole('link', { name: /DINOv3 License/i })).toHaveAttribute(
            'href',
            'https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md'
        );
    });

    it('calls onAccept when Accept and continue is pressed', async () => {
        const onAccept = vi.fn();
        renderLicense(onAccept);

        await userEvent.click(screen.getByRole('button', { name: /Accept and continue/i }));

        expect(onAccept).toHaveBeenCalledTimes(1);
    });
});
