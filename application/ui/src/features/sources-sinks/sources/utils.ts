/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    ImagesFolderSourceType,
    type Source,
    type SourcesType,
    type SourceType,
    type VideoFileSourceType,
    type WebcamSourceType,
} from '@geti-prompt/api';

const getSource = <T extends Source>(sources: SourcesType | undefined, sourceType: SourceType) => {
    return sources?.filter((source) => source.config.source_type === sourceType)[0] as T | undefined;
};

export const getWebcamSource = (sources: SourcesType | undefined) => {
    return getSource<WebcamSourceType>(sources, 'webcam');
};

export const getVideoSource = (sources: SourcesType | undefined) => {
    return getSource<VideoFileSourceType>(sources, 'video_file');
};

export const getImagesFolderSource = (sources: SourcesType | undefined) => {
    return getSource<ImagesFolderSourceType>(sources, 'images_folder');
};

export type SourcesViews = 'add' | 'edit' | 'list' | 'existing';

export const isWebcamSource = (source: Source | undefined): source is WebcamSourceType =>
    source?.config.source_type === 'webcam';

export const isImagesFolderSource = (source: Source | undefined): source is ImagesFolderSourceType =>
    source?.config.source_type === 'images_folder';
