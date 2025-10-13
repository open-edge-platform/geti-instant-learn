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

const useCreateWebcamSource = () => {
    const { projectId } = useProjectIdentifier();
    const createWebcamSourceMutation = $api.useMutation('post', '/api/v1/projects/{project_id}/sources');
    const queryClient = useQueryClient();

    const createWebcamSource = (deviceId: number) => {
        createWebcamSourceMutation.mutate(
            {
                body: {
                    id: uuid(),
                    connected: true,
                    config: {
                        source_type: 'webcam',
                        device_id: deviceId,
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
        mutate: createWebcamSource,
        isPending: createWebcamSourceMutation.isPending,
    };
};

const useUpdateWebcamSource = () => {
    const { projectId } = useProjectIdentifier();
    const updateWebcamSourceMutation = $api.useMutation('put', '/api/v1/projects/{project_id}/sources/{source_id}');
    const queryClient = useQueryClient();

    const updateWebcamSource = (sourceId: string, deviceId: number) => {
        updateWebcamSourceMutation.mutate(
            {
                body: {
                    connected: true,
                    config: {
                        source_type: 'webcam',
                        device_id: deviceId,
                    },
                },
                params: {
                    path: {
                        project_id: projectId,
                        source_id: sourceId,
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
                                    'put',
                                    `/api/v1/projects/{project_id}/sources/{source_id}`,
                                    {
                                        params: {
                                            path: {
                                                project_id: projectId,
                                                source_id: sourceId,
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
    const createWebcamSource = useCreateWebcamSource();
    const updateWebcamSource = useUpdateWebcamSource();

    const isApplyPending = createWebcamSource.isPending || updateWebcamSource.isPending;

    const isApplyDisabled =
        createWebcamSource.isPending ||
        updateWebcamSource.isPending ||
        (selectedDeviceId === source?.config?.device_id && source?.connected);

    const handleApply = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (selectedDeviceId === undefined) return;

        if (source !== undefined) {
            updateWebcamSource.mutate(source.id, parseInt(selectedDeviceId));
            return;
        }

        createWebcamSource.mutate(parseInt(selectedDeviceId));
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

            <Button marginTop={'size-200'} type={'submit'} isPending={isApplyPending} isDisabled={isApplyDisabled}>
                Apply
            </Button>
        </form>
    );
};
