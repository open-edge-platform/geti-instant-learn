/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useEffect, useState } from 'react';

import { $api, WebcamConfig } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Button, Content, Flex, Heading, InlineAlert, Loading, Radio, RadioGroup, Text, View } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { isEqual } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { CameraPermissionStatus, getAvailableVideoDevices, getVideoUserMedia } from './camera-utils';

interface WebcamSourceProps {
    source: WebcamConfig | undefined;
}

const useUpdateWebcamSource = () => {
    const { projectId } = useProjectIdentifier();
    const updateWebcamSourceMutation = $api.useMutation('post', '/api/v1/projects/{project_id}/sources');
    const queryClient = useQueryClient();

    const updateWebcamSource = (deviceId: string) => {
        updateWebcamSourceMutation.mutate(
            {
                body: {
                    id: uuid(),
                    connected: true,
                    config: {
                        source_type: 'webcam',
                        // TODO: Double check if parseInt is needed when we get the available sources from the server
                        device_id: parseInt(deviceId),
                    },
                },
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            },
            {
                onSuccess: async () => {
                    await queryClient.invalidateQueries({
                        predicate: (query) => {
                            return (
                                Array.isArray(query.queryKey) &&
                                isEqual(query.queryKey, [
                                    'get',
                                    `/api/v1/projects/{project_id}/sources`,
                                    {
                                        params: {
                                            path: {
                                                project_id: projectId,
                                            },
                                        },
                                    },
                                ])
                            );
                        },
                    });
                },
            }
        );
    };

    return {
        mutate: updateWebcamSource,
        isPending: updateWebcamSourceMutation.isPending,
    };
};

export const WebcamSource = ({ source }: WebcamSourceProps) => {
    const [availableVideoDevices, setAvailableVideoDevices] = useState<MediaDeviceInfo[] | null>(null);
    const [cameraPermission, setCameraPermission] = useState<CameraPermissionStatus>('pending');
    const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(source?.config?.device_id?.toString());
    const updateWebcamSource = useUpdateWebcamSource();
    const isApplyDisabled = updateWebcamSource.isPending || selectedDeviceId === source?.config?.device_id;

    const handleApply = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (selectedDeviceId === undefined) return;

        updateWebcamSource.mutate(selectedDeviceId);
    };

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
        <form onSubmit={handleApply}>
            <RadioGroup isEmphasized value={selectedDeviceId} onChange={changeSelectedDevice}>
                {availableVideoDevices.map((device) => (
                    <Radio key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${device.deviceId}`}
                    </Radio>
                ))}
            </RadioGroup>

            <Button
                marginTop={'size-200'}
                type={'submit'}
                isPending={updateWebcamSource.isPending}
                isDisabled={isApplyDisabled}
            >
                Apply
            </Button>
        </form>
    );
};
