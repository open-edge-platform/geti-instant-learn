/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { render } from '@/test-utils';
import { fireEvent, screen } from '@testing-library/react';

import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import { ZoomManagement } from './zoom-management.component';

const renderWithZoomProvider = (ui: ReactNode) => {
    return render(<ZoomProvider>{ui}</ZoomProvider>);
};

describe('ZoomManagement', () => {
    it('renders zoom controls', () => {
        renderWithZoomProvider(<ZoomManagement />);

        expect(screen.getByLabelText('Zoom in')).toBeInTheDocument();
        expect(screen.getByLabelText('Zoom out')).toBeInTheDocument();
        expect(screen.getByLabelText('Fit image to screen')).toBeInTheDocument();
        expect(screen.getByText(/100.0%/)).toBeInTheDocument();
    });

    it('displays current zoom percentage', () => {
        renderWithZoomProvider(<ZoomManagement />);

        const zoomText = screen.getByText(/100.0%/);
        expect(zoomText).toBeInTheDocument();
    });

    it('disables zoom in button when at maximum zoom', () => {
        renderWithZoomProvider(<ZoomManagement />);

        const zoomInButton = screen.getByLabelText('Zoom in');

        for (let i = 0; i < 20; i++) {
            if (!zoomInButton.hasAttribute('disabled')) {
                fireEvent.click(zoomInButton);
            }
        }

        expect(zoomInButton).toHaveAttribute('disabled');
    });

    it('zoom out button is disabled at initial zoom level', () => {
        renderWithZoomProvider(<ZoomManagement />);

        const zoomOutButton = screen.getByLabelText('Zoom out');

        expect(zoomOutButton).toHaveAttribute('disabled');
    });
});
