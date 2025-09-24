/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { uniqBy } from 'lodash-es';

export type CameraPermissionStatus = 'granted' | 'denied' | 'pending';

export const isVideoInput = (mediaDevice: MediaDeviceInfo) => {
    return mediaDevice.kind === 'videoinput';
};

export const getVideoUserMedia = (constraints: MediaStreamConstraints = { video: true }) => {
    return navigator.mediaDevices.getUserMedia(constraints);
};

export const getVideoDevices = async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();

    const videoDevices = devices.filter(isVideoInput);

    return uniqBy(videoDevices, (device) => device.deviceId);
};

export const getBrowserPermission = async (): Promise<
    { stream: MediaStream; permission: 'granted' } | { stream: null; permission: 'denied' }
> => {
    try {
        const stream = await getVideoUserMedia();

        return { permission: 'granted', stream };
    } catch {
        return { permission: 'denied', stream: null };
    }
};

export const getAvailableVideoDevices = async () => {
    const { permission, stream } = await getBrowserPermission();

    if (permission === 'granted') {
        const devices = await getVideoDevices();

        return {
            devices,
            stream,
            permission,
        };
    } else {
        return {
            devices: null,
            stream: null,
            permission,
        };
    }
};
