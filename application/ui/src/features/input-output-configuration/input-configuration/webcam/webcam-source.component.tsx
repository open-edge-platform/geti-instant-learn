/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from 'react';

import { Content, Flex, Heading, InlineAlert, Loading, Radio, RadioGroup, Text, View } from '@geti/ui';

import { CameraPermissionStatus, getAvailableVideoDevices, getVideoUserMedia } from './camera-utils';

export const WebcamSource = () => {
    const [availableVideoDevices, setAvailableVideoDevices] = useState<MediaDeviceInfo[] | null>(null);
    const [cameraPermission, setCameraPermission] = useState<CameraPermissionStatus>('pending');
    const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(undefined);

    const changeSelectedDevice = async (deviceId: string) => {
        setSelectedDeviceId(deviceId);
        await getVideoUserMedia({
            video: {
                deviceId: { exact: deviceId },
            },
        });
    };

    useEffect(() => {
        const updateDevices = async () => {
            const { devices, permission } = await getAvailableVideoDevices();
            setAvailableVideoDevices(devices);
            setCameraPermission(permission);
            if (devices?.length) setSelectedDeviceId(devices[0].deviceId);
        };

        updateDevices();

        navigator.mediaDevices.addEventListener('devicechange', updateDevices);
        return () => navigator.mediaDevices.removeEventListener('devicechange', updateDevices);
    }, []);

    if (cameraPermission === 'pending') {
        return (
            <View>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Loading mode={'inline'} size={'S'} />
                    <Text>Checking browser camera permission...</Text>
                </Flex>
            </View>
        );
    }

    if (cameraPermission === 'denied' || availableVideoDevices === null) {
        return (
            <InlineAlert variant='notice'>
                <Heading>Update camera permission</Heading>
                <Content>Please grant camera permission to be able to use webcam as input source.</Content>
            </InlineAlert>
        );
    }

    return (
        <RadioGroup isEmphasized value={selectedDeviceId} onChange={changeSelectedDevice}>
            {availableVideoDevices.map((device) => (
                <Radio key={device.deviceId} value={device.deviceId}>
                    {device.label || `Camera ${device.deviceId}`}
                </Radio>
            ))}
        </RadioGroup>
    );
};
