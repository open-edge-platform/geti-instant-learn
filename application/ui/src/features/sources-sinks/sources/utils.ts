/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    ImagesFolderSourceType,
    SampleDatasetSourceType,
    type Source,
    type SourcesType,
    type SourceType,
    type USBCameraSourceType,
    type VideoFileSourceType,
} from '@geti-prompt/api';

const getSource = <T extends Source>(sources: SourcesType | undefined, sourceType: SourceType) => {
    return sources?.filter((source) => source.config.source_type === sourceType)[0] as T | undefined;
};

export const getUsbCameraSource = (sources: SourcesType | undefined) => {
    return getSource<USBCameraSourceType>(sources, 'usb_camera');
};

export const getVideoSource = (sources: SourcesType | undefined) => {
    return getSource<VideoFileSourceType>(sources, 'video_file');
};

export const getImagesFolderSource = (sources: SourcesType | undefined) => {
    return getSource<ImagesFolderSourceType>(sources, 'images_folder');
};

export type SourcesViews = 'add' | 'edit' | 'list' | 'existing';

export const isUsbCameraSource = (source: Source | undefined): source is USBCameraSourceType =>
    source?.config.source_type === 'usb_camera';

export const isImagesFolderSource = (source: Source | undefined): source is ImagesFolderSourceType =>
    source?.config.source_type === 'images_folder';

export const isTestDatasetSource = (source: Source | undefined): source is SampleDatasetSourceType =>
    source?.config.source_type === 'sample_dataset';

export const isVideoFileSource = (source: Source | undefined): source is VideoFileSourceType =>
    source?.config.source_type === 'video_file';
