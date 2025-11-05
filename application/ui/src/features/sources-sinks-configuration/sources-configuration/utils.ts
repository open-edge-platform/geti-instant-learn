/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    ImagesFolderSourceType,
    type Source,
    type SourcesListType,
    type SourceType,
    type VideoFileSourceType,
    type WebcamSourceType,
} from '@geti-prompt/api';

const getSource = <T extends Source>(sources: SourcesListType | undefined, sourceType: SourceType) => {
    return sources?.filter((source) => source.config.source_type === sourceType)[0] as T | undefined;
};

export const getWebcamSource = (sources: SourcesListType | undefined) => {
    return getSource<WebcamSourceType>(sources, 'webcam');
};

export const getVideoSource = (sources: SourcesListType | undefined) => {
    return getSource<VideoFileSourceType>(sources, 'video_file');
};

export const getImagesFolderSource = (sources: SourcesListType | undefined) => {
    return getSource<ImagesFolderSourceType>(sources, 'images_folder');
};
