/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType, SourcesListType, USBCameraSourceType, VideoFileSourceType } from '@/api';
import { getMockedImagesFolderSource, getMockedVideoFileSource, render } from '@/test-utils';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http, server } from '../../../setup-test';
import { Sources } from './sources.component';

const VIDEO_SOURCE: VideoFileSourceType = {
    id: 'video-source-id',
    active: true,
    config: { source_type: 'video_file', video_path: '/home/user/video.mp4', seekable: true },
};

const IMAGES_SOURCE: ImagesFolderSourceType = {
    id: 'images-source-id',
    active: true,
    config: { source_type: 'images_folder', images_folder_path: '/home/user/images', seekable: true },
};

const USB_SOURCE: USBCameraSourceType = {
    id: 'usb-source-id',
    active: false,
    config: { source_type: 'usb_camera', device_id: 0, name: 'Webcam HD', seekable: false },
};

const DATASET_SOURCE = {
    id: 'dataset-source-id',
    active: true,
    config: { source_type: 'sample_dataset' as const, dataset_id: 'dataset-1', seekable: true },
};

const DATASETS_RESPONSE = {
    datasets: [{ id: 'dataset-1', name: 'Sample Dataset 1', thumbnail: null }],
    pagination: { count: 1, total: 1, offset: 0, limit: 10 },
};

const mockSourcesResponse = (sources: SourcesListType['sources']): SourcesListType => ({
    sources,
    pagination: { count: sources.length, total: sources.length, offset: 0, limit: 10 },
});

describe('Sources', () => {
    beforeEach(() => {
        // Sources always prefetches available USB cameras; provide a stable mock to avoid
        // in-flight requests causing libuv stream teardown issues in jsdom
        server.use(
            http.get('/api/v1/system/source-types/{source_type}/sources', ({ response }) => {
                return response(200).json([]);
            }),
            http.get('/api/v1/system/datasets', () => {
                return HttpResponse.json({ datasets: [], pagination: { count: 0, total: 0, offset: 0, limit: 10 } });
            })
        );
    });
    describe('Initial state (no sources)', () => {
        it('shows USB Camera, Image folder, and Video file source type options', async () => {
            render(<Sources />);

            expect(await screen.findByRole('button', { name: 'USB Camera' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Image folder' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Video file' })).toBeInTheDocument();
        });

        it('does not show Sample dataset option when no datasets are available', async () => {
            render(<Sources />);

            // Wait for the component to finish loading
            await screen.findByRole('button', { name: 'USB Camera' });

            expect(screen.queryByRole('button', { name: 'Sample dataset' })).not.toBeInTheDocument();
        });
    });

    describe('Existing sources list', () => {
        it('hides the Add new source button when all source types are already present', async () => {
            server.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(
                        mockSourcesResponse([VIDEO_SOURCE, IMAGES_SOURCE, USB_SOURCE, DATASET_SOURCE])
                    );
                }),
                http.get('/api/v1/system/datasets', () => {
                    return HttpResponse.json(DATASETS_RESPONSE);
                })
            );

            render(<Sources />);

            // Wait for the existing sources list to appear
            await screen.findByTestId('pipeline-entity-card-video-file');

            expect(screen.queryByRole('button', { name: 'Add new source' })).not.toBeInTheDocument();
        });

        it('hides already existing source type from Add new source options', async () => {
            server.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([VIDEO_SOURCE]));
                })
            );

            render(<Sources />);

            // Wait for the existing sources list and click Add new source
            await userEvent.click(await screen.findByRole('button', { name: 'Add new source' }));

            // Video file should be hidden since it already exists
            expect(screen.queryByRole('button', { name: 'Video file' })).not.toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Image folder' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'USB Camera' })).toBeInTheDocument();
        });
    });

    describe('Delete source', () => {
        it('keeps the existing list visible when one of multiple sources is deleted', async () => {
            const videoSource = getMockedVideoFileSource({ id: 'video-id', active: true, filePath: '/video.mp4' });
            const imagesSource = getMockedImagesFolderSource({ active: true, imagesFolderPath: '/images' });

            let deleteCount = 0;

            server.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    if (deleteCount === 0) {
                        return response(200).json(mockSourcesResponse([videoSource, imagesSource]));
                    }
                    return response(200).json(mockSourcesResponse([imagesSource]));
                }),
                http.delete('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) => {
                    deleteCount++;
                    return response(204).empty();
                })
            );

            render(<Sources />);

            await screen.findByTestId('pipeline-entity-card-video-file');

            const videoCard = screen.getByTestId('pipeline-entity-card-video-file');
            await userEvent.click(within(videoCard).getByRole('button'));
            await userEvent.click(screen.getByRole('menuitem', { name: 'Delete' }));

            await waitFor(() => {
                expect(screen.queryByTestId('pipeline-entity-card-video-file')).not.toBeInTheDocument();
            });

            expect(screen.getByTestId('pipeline-entity-card-images-folder')).toBeInTheDocument();
        });
    });
});
