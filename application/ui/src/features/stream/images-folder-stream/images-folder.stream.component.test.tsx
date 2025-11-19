/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen } from '@testing-library/react';
import { SelectedFrameProvider } from 'src/shared/selected-frame-provider.component';

import { paths } from '../../../constants/paths';
import { WebRTCConnectionProvider } from '../web-rtc/web-rtc-connection-provider';
import { ImagesFolderStream } from './images-folder-stream.component';

describe('ImagesFolderStream', () => {
    const renderImagesFolderStream = (mode = 'visual') => {
        return render(
            <SelectedFrameProvider>
                <WebRTCConnectionProvider>
                    <ImagesFolderStream />
                </WebRTCConnectionProvider>
            </SelectedFrameProvider>,
            {
                route: `/projects/1?mode=${mode}`,
                path: paths.project.pattern,
            }
        );
    };

    it('renders frames list with all frames', () => {
        renderImagesFolderStream();

        expect(screen.getAllByAltText(/Frame/)).toHaveLength(18);
    });

    it('renders capture frame button when in visual prompt mode', () => {
        renderImagesFolderStream('visual');

        expect(screen.getByRole('button', { name: 'Capture' })).toBeInTheDocument();
    });

    it('does not render capture frame button when not in visual prompt mode', () => {
        renderImagesFolderStream('text');

        expect(screen.queryByRole('button', { name: 'Capture' })).not.toBeInTheDocument();
    });

    it('disables previous button when on first frame', () => {
        renderImagesFolderStream();

        expect(screen.getByRole('button', { name: 'Previous Frame' })).toBeDisabled();
    });

    it('disables next button when on last frame', () => {
        renderImagesFolderStream();

        fireEvent.click(screen.getByLabelText('Frame #17'));

        expect(screen.getByRole('button', { name: 'Next Frame' })).toBeDisabled();
    });

    it('navigates to next/previous frame when next button is clicked', () => {
        renderImagesFolderStream();

        const frame0 = screen.getByRole('option', { name: 'Frame #0' });
        expect(frame0).toHaveAttribute('data-isSelected', 'true');

        fireEvent.click(screen.getByRole('button', { name: 'Next Frame' }));

        expect(screen.getByRole('option', { name: 'Frame #1' })).toHaveAttribute('data-isSelected', 'true');

        fireEvent.click(screen.getByRole('button', { name: 'Previous Frame' }));

        expect(screen.getByRole('option', { name: 'Frame #0' })).toHaveAttribute('data-isSelected', 'true');
    });

    it('supports keyboard navigation with arrow keys', () => {
        renderImagesFolderStream();

        const frame0 = screen.getByRole('option', { name: 'Frame #0' });
        expect(frame0).toHaveAttribute('data-isSelected', 'true');

        fireEvent.keyDown(document, { key: 'ArrowRight' });

        const frame1 = screen.getByRole('option', { name: 'Frame #1' });
        expect(frame1).toHaveAttribute('data-isSelected', 'true');

        // Simulate ArrowLeft key press
        fireEvent.keyDown(document, { key: 'ArrowLeft' });

        expect(frame0).toHaveAttribute('data-isSelected', 'true');
    });

    it('does not navigate beyond first frame when pressing ArrowLeft', () => {
        renderImagesFolderStream();

        const frame0 = screen.getByRole('option', { name: 'Frame #0' });
        expect(frame0).toHaveAttribute('data-isSelected', 'true');

        // Try to navigate left from first frame
        fireEvent.keyDown(document, { key: 'ArrowLeft' });

        expect(frame0).toHaveAttribute('data-isSelected', 'true');
    });

    it('does not navigate beyond last frame when pressing ArrowRight', () => {
        renderImagesFolderStream();

        // Navigate to last frame
        const frame17 = screen.getByRole('option', { name: 'Frame #17' });
        fireEvent.click(frame17);

        expect(frame17).toHaveAttribute('data-isSelected', 'true');

        // Try to navigate right from last frame
        fireEvent.keyDown(document, { key: 'ArrowRight' });

        expect(frame17).toHaveAttribute('data-isSelected', 'true');
    });
});
