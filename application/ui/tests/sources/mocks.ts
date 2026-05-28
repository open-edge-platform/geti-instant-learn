/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    ImagesFolderSourceType,
    SampleDatasetSourceType,
    SourcesListType,
    USBCameraSourceType,
    VideoFileSourceType,
} from '../../src/api';

export const VIDEO_SOURCE: VideoFileSourceType = {
    id: 'video-source-id',
    active: true,
    config: { source_type: 'video_file', video_path: '/home/user/video.mp4', seekable: true },
};

export const IMAGES_SOURCE: ImagesFolderSourceType = {
    id: 'images-source-id',
    active: true,
    config: { source_type: 'images_folder', images_folder_path: '/home/user/images', seekable: true },
};

export const USB_SOURCE: USBCameraSourceType = {
    id: 'usb-source-id',
    active: false,
    config: { source_type: 'usb_camera', device_id: 0, name: 'Webcam HD', seekable: false },
};

export const DATASET_SOURCE: SampleDatasetSourceType = {
    id: 'dataset-source-id',
    active: true,
    config: { source_type: 'sample_dataset', dataset_id: 'dataset-1', seekable: true },
};

export const mockSourcesResponse = (sources: SourcesListType['sources']): SourcesListType => ({
    sources,
    pagination: { count: sources.length, total: sources.length, offset: 0, limit: 10 },
});

// The available-sources endpoint returns an array of config objects with source_type discriminator
export const USB_CAMERAS_RESPONSE = [
    { source_type: 'usb_camera' as const, device_id: 0, name: 'Webcam HD', seekable: false },
];

export const DATASETS_RESPONSE = {
    datasets: [
        { id: 'dataset-1', name: 'Sample Dataset 1', thumbnail: null },
        { id: 'dataset-2', name: 'Sample Dataset 2', thumbnail: null },
    ],
    pagination: { count: 2, total: 2, offset: 0, limit: 10 },
};

export const EMPTY_DATASETS_RESPONSE = {
    datasets: [],
    pagination: { count: 0, total: 0, offset: 0, limit: 10 },
};
