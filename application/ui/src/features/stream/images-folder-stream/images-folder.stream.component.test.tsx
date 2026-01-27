/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameAPIType } from '@/api';
import { render } from '@/test-utils';
import { fireEvent, screen, waitForElementToBeRemoved } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { SelectedFrameProvider } from 'src/shared/selected-frame-provider.component';
import { beforeEach } from 'vitest';

import { paths } from '../../../constants/paths';
import { http, server } from '../../../setup-test';
import { FullScreenModeProvider } from '../../annotator/actions/full-screen-mode.component';
import { WebRTCConnectionProvider } from '../web-rtc/web-rtc-connection-provider';
import { ImagesFolderStream } from './images-folder-stream.component';

const getMockedFrame = (frame: Partial<FrameAPIType> = {}): FrameAPIType => {
    return {
        index: 0,
        thumbnail: '',
        ...frame,
    };
};

const renderImagesFolderStream = async (mode = 'visual', sourceId = '1234') => {
    render(
        <SelectedFrameProvider>
            <FullScreenModeProvider>
                <WebRTCConnectionProvider>
                    <ImagesFolderStream sourceId={sourceId} />
                </WebRTCConnectionProvider>
            </FullScreenModeProvider>
        </SelectedFrameProvider>,
        {
            route: `/projects/1?mode=${mode}`,
            path: paths.project.pattern,
        }
    );

    await waitForElementToBeRemoved(screen.getByRole('progressbar'));
};

describe('ImagesFolderStream', () => {
    beforeEach(() => {
        let activeIndex = 0;

        server.use(
            http.get('/api/v1/projects/{project_id}/sources/{source_id}/frames', () => {
                const frames = Array.from({ length: 18 }).map((_, index) =>
                    getMockedFrame({
                        index,
                        thumbnail: `base64;frame-${index}`,
                    })
                );

                return HttpResponse.json({
                    frames,
                    pagination: {
                        count: 18,
                        total: 18,
                        offset: 0,
                        limit: 20,
                    },
                });
            }),

            http.post('/api/v1/projects/{project_id}/sources/{source_id}/frames/{index}', ({ params }) => {
                activeIndex = Number(params.index);

                return HttpResponse.json({
                    index: activeIndex,
                });
            }),

            http.get('/api/v1/projects/{project_id}/sources/{source_id}/frames/index', () => {
                return HttpResponse.json({
                    index: activeIndex,
                });
            })
        );
    });

    it('renders frames list with all frames', async () => {
        await renderImagesFolderStream();

        expect(await screen.findAllByAltText(/Frame/)).toHaveLength(18);
    });

    it('renders capture frame button when in visual prompt mode', async () => {
        await renderImagesFolderStream('visual');

        expect(screen.getByRole('button', { name: 'Capture' })).toBeInTheDocument();
    });

    // TODO: Uncomment when we support text prompt
    it.skip('does not render capture frame button when not in visual prompt mode', async () => {
        await renderImagesFolderStream('text');

        expect(screen.queryByRole('button', { name: 'Capture' })).not.toBeInTheDocument();
    });

    it('disables previous button when on first frame', async () => {
        await renderImagesFolderStream();

        expect(await screen.findByRole('button', { name: 'Previous Frame' })).toBeDisabled();
    });

    it('disables next button when on last frame', async () => {
        await renderImagesFolderStream();

        fireEvent.click(await screen.findByLabelText('Frame #17'));

        expect(await screen.findByRole('button', { name: 'Next Frame' })).toBeDisabled();
    });

    it('navigates to next/previous frame when next button is clicked', async () => {
        await renderImagesFolderStream();

        const frame0 = await screen.findByRole('option', { name: 'Frame #0' });
        expect(frame0).toHaveAttribute('data-isselected', 'true');

        fireEvent.click(screen.getByRole('button', { name: 'Next Frame' }));

        expect(await screen.findByRole('option', { name: 'Frame #1' })).toHaveAttribute('data-isselected', 'true');

        fireEvent.click(screen.getByRole('button', { name: 'Previous Frame' }));

        expect(await screen.findByRole('option', { name: 'Frame #0' })).toHaveAttribute('data-isselected', 'true');
    });

    it('supports keyboard navigation with arrow keys', async () => {
        await renderImagesFolderStream();

        const frame0 = await screen.findByRole('option', { name: 'Frame #0' });
        expect(frame0).toHaveAttribute('data-isselected', 'true');

        fireEvent.keyDown(document, { key: 'ArrowRight' });

        const frame1 = await screen.findByRole('option', { name: 'Frame #1' });
        expect(frame1).toHaveAttribute('data-isselected', 'true');

        // Simulate ArrowLeft key press
        fireEvent.keyDown(document, { key: 'ArrowLeft' });

        expect(await screen.findByRole('option', { name: 'Frame #0' })).toHaveAttribute('data-isselected', 'true');
    });

    it('does not navigate beyond first frame when pressing ArrowLeft', async () => {
        await renderImagesFolderStream();

        expect(await screen.findByRole('option', { name: 'Frame #0' })).toHaveAttribute('data-isselected', 'true');

        // Try to navigate left from first frame
        fireEvent.keyDown(document, { key: 'ArrowLeft' });

        expect(await screen.findByRole('option', { name: 'Frame #0' })).toHaveAttribute('data-isselected', 'true');
    });

    it('does not navigate beyond last frame when pressing ArrowRight', async () => {
        await renderImagesFolderStream();

        // Navigate to last frame
        fireEvent.click(await screen.findByRole('option', { name: 'Frame #17' }));

        expect(await screen.findByRole('option', { name: 'Frame #17' })).toHaveAttribute('data-isselected', 'true');

        // Try to navigate right from last frame
        fireEvent.keyDown(document, { key: 'ArrowRight' });

        expect(await screen.findByRole('option', { name: 'Frame #17' })).toHaveAttribute('data-isselected', 'true');
    });
});
