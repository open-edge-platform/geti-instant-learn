/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    ImagesFolderConfig,
    type SourceConfig,
    type SourcesListType,
    type SourceType,
    type VideoFileConfig,
    type WebcamConfig,
} from '@geti-prompt/api';

const getSource = <T extends SourceConfig>(sources: SourcesListType | undefined, sourceType: SourceType) => {
    return sources?.filter((source) => source.config.source_type === sourceType)[0] as T | undefined;
};

export const getWebcamSource = (sources: SourcesListType | undefined) => {
    return getSource<WebcamConfig>(sources, 'webcam');
};

export const getVideoSource = (sources: SourcesListType | undefined) => {
    return getSource<VideoFileConfig>(sources, 'video_file');
};

export const getImagesFolderSource = (sources: SourcesListType | undefined) => {
    return getSource<ImagesFolderConfig>(sources, 'images_folder');
};
