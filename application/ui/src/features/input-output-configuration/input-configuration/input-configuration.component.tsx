/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { GenICam, ImagesFolder, IPCamera, VideoFile, WebCam } from '@geti-prompt/icons';

import { DisclosureGroup } from '../ui/disclosure-group/disclosure-group.component';
import { IPCameraForm } from './ip-camera.component';
import { WebcamSource } from './webcam/webcam-source.component';

const inputs = [
    { label: 'Webcam', value: 'webcam', content: <WebcamSource />, icon: <WebCam width={'24px'} />, isActive: true },
    {
        label: 'IP Camera',
        value: 'ip-camera',
        content: <IPCameraForm />,
        icon: <IPCamera width={'24px'} />,
        isActive: false,
    },
    { label: 'GenICam', value: 'gen-i-cam', content: 'Test', icon: <GenICam width={'24px'} />, isActive: false },
    { label: 'Video file', value: 'video-file', content: 'Test', icon: <VideoFile width={'24px'} />, isActive: false },
    {
        label: 'Image folder',
        value: 'image-folder',
        content: 'Test',
        icon: <ImagesFolder width={'24px'} />,
        isActive: false,
    },
];

export const InputConfiguration = () => {
    const activeInput = inputs.find((input) => input.isActive)?.value ?? null;

    return <DisclosureGroup items={inputs} value={activeInput} />;
};
