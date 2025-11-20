/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { SourceType } from '@geti-prompt/api';
import { useGetSources } from '@geti-prompt/hooks';
import { ImagesFolder as ImagesFolderIcon, WebCam } from '@geti-prompt/icons';

import { DisclosureGroup } from '../disclosure-group/disclosure-group.component';
import { ImagesFolder } from './images-folder/images-folder.component';
import { getImagesFolderSource, getWebcamSource } from './utils';
import { WebcamSource } from './webcam/webcam-source.component';

export const Sources = () => {
    const { data } = useGetSources();

    const sources: {
        label: string;
        value: SourceType;
        content: ReactNode;
        icon: ReactNode;
    }[] = [
        {
            label: 'Webcam',
            value: 'webcam',
            content: <WebcamSource source={getWebcamSource(data?.sources)} />,
            icon: <WebCam width={'24px'} />,
        },
        /*{
            label: 'IP Camera',
            value: 'ip_camera',
            content: <IPCameraForm />,
            icon: <IPCamera width={'24px'} />,
        },*/
        /*{ label: 'GenICam', value: 'gen-i-cam', content: 'Test', icon: <GenICam width={'24px'} /> },*/
        /*{
            label: 'Video file',
            value: 'video_file',
            content: 'Test',
            icon: <VideoFile width={'24px'} />,
        },*/
        {
            label: 'Image folder',
            value: 'images_folder',
            content: <ImagesFolder source={getImagesFolderSource(data?.sources)} />,
            icon: <ImagesFolderIcon width={'24px'} />,
        },
    ];

    const activeSource = data.sources.find((source) => source.connected)?.config.source_type;

    return <DisclosureGroup items={sources} value={activeSource} />;
};
