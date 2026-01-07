/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SourceUpdateType, VideoFileSourceType } from '@geti-prompt/api';
import { getMockedVideoFileSource, render } from '@geti-prompt/test-utils';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, vi } from 'vitest';

import { http } from '../../../../api/utils';
import { server } from '../../../../setup-test';
import { EditVideoFile } from './edit-video-file.component';

class VideoFilePage {
    constructor() {}

    get filePathField() {
        return screen.getByRole('textbox', { name: /File path/ });
    }

    get saveButton() {
        return screen.getByRole('button', { name: 'Save' });
    }

    get saveAndConnectButton() {
        return screen.queryByRole('button', { name: 'Save & Connect' });
    }

    async save() {
        await userEvent.click(this.saveButton);
    }

    async saveAndConnect() {
        const btn = this.saveAndConnectButton;
        if (btn) {
            await userEvent.click(this.saveAndConnectButton);
        }
    }

    async setFilePath(path: string) {
        await userEvent.clear(this.filePathField);
        await userEvent.type(this.filePathField, path);
    }
}

const renderVideoFile = ({ source, onSaved = vi.fn() }: { source: VideoFileSourceType; onSaved?: () => void }) => {
    const result = render(<EditVideoFile source={source} onSaved={onSaved} />);

    return {
        result,
        videoFilePage: new VideoFilePage(),
    };
};

describe('EditVideoFile', () => {
    describe('Active source', () => {
        const activeSource = getMockedVideoFileSource({ active: true });

        it('displays only save button', () => {
            const { videoFilePage } = renderVideoFile({ source: activeSource });

            expect(videoFilePage.saveButton).toBeInTheDocument();
            expect(videoFilePage.saveAndConnectButton).not.toBeInTheDocument();
        });

        it('disables save button when path is empty', () => {
            const source = getMockedVideoFileSource({ active: true, filePath: '' });
            const { videoFilePage } = renderVideoFile({ source });

            expect(videoFilePage.filePathField).toHaveValue('');
            expect(videoFilePage.saveButton).toBeDisabled();
        });

        it('disables save button when path is equal to source path', () => {
            const source = getMockedVideoFileSource({ filePath: '/path/to/file.mp4', active: true });

            const { videoFilePage } = renderVideoFile({ source });

            expect(videoFilePage.filePathField).toHaveValue(source.config.video_path);
            expect(videoFilePage.saveButton).toBeDisabled();
        });

        it('enables submit button when path is different from source path', async () => {
            const videoFilePath = '/path/to/another_file.mp4';
            const source = getMockedVideoFileSource({ filePath: '/path/to/file.mp4', active: true });

            const { videoFilePage } = renderVideoFile({ source });

            expect(videoFilePage.filePathField).toHaveValue(source.config.video_path);
            expect(videoFilePage.saveButton).toBeDisabled();

            await videoFilePage.setFilePath(videoFilePath);

            expect(videoFilePage.filePathField).toHaveValue(videoFilePath);
            expect(videoFilePage.saveButton).toBeEnabled();
        });

        it('updates source with provided path', async () => {
            const videoFilePath = '/path/to/another_file.mp4';
            const source = getMockedVideoFileSource({ filePath: '/path/to/file.mp4', active: true });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();

                    return HttpResponse.json({}, { status: 200 });
                })
            );

            const { videoFilePage } = renderVideoFile({ source });

            await videoFilePage.setFilePath(videoFilePath);
            await videoFilePage.save();

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
        });
    });

    describe('Inactive source', () => {
        const inactiveSource = getMockedVideoFileSource({ active: false });

        it('displays save and save&connect buttons', () => {
            const { videoFilePage } = renderVideoFile({ source: inactiveSource });

            expect(videoFilePage.saveButton).toBeInTheDocument();
            expect(videoFilePage.saveAndConnectButton).toBeInTheDocument();
        });

        it('disables save buttons when path is empty', () => {
            const { videoFilePage } = renderVideoFile({ source: inactiveSource });

            expect(videoFilePage.filePathField).toHaveValue('');
            expect(videoFilePage.saveButton).toBeDisabled();
            expect(videoFilePage.saveAndConnectButton).toBeDisabled();
        });

        it('disables save buttons when path is equal to source path', () => {
            const source = getMockedVideoFileSource({ filePath: '/path/to/file.mp4', active: false });

            const { videoFilePage } = renderVideoFile({ source });

            expect(videoFilePage.filePathField).toHaveValue(source.config.video_path);
            expect(videoFilePage.saveButton).toBeDisabled();
            expect(videoFilePage.saveAndConnectButton).toBeDisabled();
        });

        it('enables submit button when path is different from source path', async () => {
            const anotherFilePath = '/path/to/another_file.mp4';
            const source = getMockedVideoFileSource({ filePath: '/path/to/file.mp4', active: false });

            const { videoFilePage } = renderVideoFile({ source });

            expect(videoFilePage.filePathField).toHaveValue(source.config.video_path);
            expect(videoFilePage.saveButton).toBeDisabled();
            expect(videoFilePage.saveAndConnectButton).toBeDisabled();

            await videoFilePage.setFilePath(anotherFilePath);

            expect(videoFilePage.filePathField).toHaveValue(anotherFilePath);
            expect(videoFilePage.saveButton).toBeEnabled();
            expect(videoFilePage.saveAndConnectButton).toBeEnabled();
        });

        it('updates source with provided path', async () => {
            const anotherFilePath = '/path/to/another_file.mp4';
            const source = getMockedVideoFileSource({ filePath: '/path/to/file.mp4', active: false });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();

                    return HttpResponse.json({}, { status: 200 });
                })
            );

            const { videoFilePage } = renderVideoFile({ source });

            await videoFilePage.setFilePath(anotherFilePath);
            await videoFilePage.save();

            await waitFor(() => {
                expect(body).toEqual(
                    expect.objectContaining({
                        active: false,
                        config: {
                            seekable: true,
                            source_type: 'video_file',
                            video_path: anotherFilePath,
                        },
                    })
                );
            });
        });

        it('updates source with provided path and connect', async () => {
            const anotherFilePath = '/path/to/another_file.mp4';

            const source = getMockedVideoFileSource({ filePath: '/path/to/file.mp4', active: false });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();

                    return HttpResponse.json({}, { status: 200 });
                })
            );

            const { videoFilePage } = renderVideoFile({ source });

            await videoFilePage.setFilePath(anotherFilePath);
            await videoFilePage.saveAndConnect();

            await waitFor(() => {
                expect(body).toEqual(
                    expect.objectContaining({
                        active: true,
                        config: {
                            seekable: true,
                            source_type: 'video_file',
                            video_path: anotherFilePath,
                        },
                    })
                );
            });
        });
    });
});
