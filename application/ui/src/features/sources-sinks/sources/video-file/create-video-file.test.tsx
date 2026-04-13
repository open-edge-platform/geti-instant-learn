/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SourceCreateType } from '@/api';
import { render } from '@/test-utils';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http } from '../../../../api/utils';
import { server } from '../../../../setup-test';
import { CreateVideoFile } from './create-video-file.component';

class VideoFilePage {
    constructor() {}

    get filePathField() {
        return screen.getByRole('textbox', { name: /File path/ });
    }

    get browseButton() {
        return screen.queryByRole('button', { name: 'Browse' });
    }

    get applyButton() {
        return screen.getByRole('button', { name: 'Apply' });
    }

    async submit() {
        await userEvent.click(this.applyButton);
    }

    async setFilePath(path: string) {
        await userEvent.clear(this.filePathField);
        await userEvent.type(this.filePathField, path);
    }
}

const renderVideoFile = (onSaved = vi.fn()) => {
    const result = render(<CreateVideoFile onSaved={onSaved} />);

    return {
        result,
        videoFilePage: new VideoFilePage(),
    };
};
/* eslint-disable no-underscore-dangle */
const setTauriContext = () => {
    window.__TAURI__ = {
        core: {
            invoke: vi.fn(),
        },
    };
};

afterEach(() => {
    window.__TAURI__ = undefined;
});
/* eslint-enable no-underscore-dangle */

describe('CreateVideoFile', () => {
    it('shows browse button in tauri context', () => {
        setTauriContext();

        const { videoFilePage } = renderVideoFile();

        expect(videoFilePage.browseButton).toBeInTheDocument();
    });

    it('disables apply button when file path is empty', () => {
        const { videoFilePage } = renderVideoFile();

        expect(videoFilePage.filePathField).toHaveValue('');
        expect(videoFilePage.browseButton).not.toBeInTheDocument();
        expect(videoFilePage.applyButton).toBeDisabled();
    });

    it('creates video file source when file path is valid', async () => {
        const videoFilePath = '/path/to/video/file.mp4';
        const mockOnSaved = vi.fn();

        let body: SourceCreateType | null = null;

        server.use(
            http.post('/api/v1/projects/{project_id}/sources', async ({ request }) => {
                body = await request.json();

                return HttpResponse.json({}, { status: 201 });
            })
        );

        const { videoFilePage } = renderVideoFile(mockOnSaved);

        expect(videoFilePage.applyButton).toBeDisabled();

        await videoFilePage.setFilePath(videoFilePath);

        expect(videoFilePage.applyButton).toBeEnabled();

        await videoFilePage.submit();

        await waitFor(() => {
            expect(body).toEqual(
                expect.objectContaining({
                    active: true,
                    config: {
                        seekable: true,
                        source_type: 'video_file',
                        video_path: videoFilePath,
                    },
                })
            );
        });

        expect(mockOnSaved).toHaveBeenCalled();
    });
});
