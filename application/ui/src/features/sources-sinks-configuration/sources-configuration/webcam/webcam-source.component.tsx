/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { $api, WebcamConfig } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Button, TextField, View } from '@geti/ui';
import { isInteger } from 'lodash-es';
import { v4 as uuid } from 'uuid';

interface WebcamSourceProps {
    source: WebcamConfig | undefined;
}

const useCreateWebcamSource = () => {
    const { projectId } = useProjectIdentifier();
    const createWebcamSourceMutation = $api.useMutation('post', '/api/v1/projects/{project_id}/sources', {
        meta: {
            invalidates: [
                [
                    'get',
                    '/api/v1/projects/{project_id}/sources',
                    {
                        params: {
                            path: {
                                project_id: projectId,
                            },
                        },
                    },
                ],
            ],
        },
    });

    const createWebcamSource = (deviceId: number) => {
        createWebcamSourceMutation.mutate({
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
        });
    };

    return {
        mutate: createWebcamSource,
        isPending: createWebcamSourceMutation.isPending,
    };
};

const useUpdateWebcamSource = () => {
    const { projectId } = useProjectIdentifier();
    const updateWebcamSourceMutation = $api.useMutation('put', '/api/v1/projects/{project_id}/sources/{source_id}', {
        meta: {
            invalidates: [
                [
                    'get',
                    '/api/v1/projects/{project_id}/sources',
                    {
                        params: {
                            path: {
                                project_id: projectId,
                            },
                        },
                    },
                ],
            ],
        },
    });

    const updateWebcamSource = (sourceId: string, deviceId: number) => {
        updateWebcamSourceMutation.mutate({
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
        });
    };

    return {
        mutate: updateWebcamSource,
        isPending: updateWebcamSourceMutation.isPending,
    };
};

export const WebcamSource = ({ source }: WebcamSourceProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(
        source?.config?.device_id?.toString() ?? '0'
    );
    const createWebcamSource = useCreateWebcamSource();
    const updateWebcamSource = useUpdateWebcamSource();

    const isApplyPending = createWebcamSource.isPending || updateWebcamSource.isPending;

    const isApplyDisabled =
        createWebcamSource.isPending ||
        updateWebcamSource.isPending ||
        (selectedDeviceId == source?.config?.device_id && source?.connected);

    const handleApply = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (selectedDeviceId === undefined) return;

        if (source !== undefined) {
            updateWebcamSource.mutate(source.id, parseInt(selectedDeviceId));
            return;
        }

        createWebcamSource.mutate(parseInt(selectedDeviceId));
    };

    return (
        <form onSubmit={handleApply}>
            <View>
                <TextField
                    label={'Device ID'}
                    value={selectedDeviceId}
                    onChange={setSelectedDeviceId}
                    validate={(value) => (isInteger(Number(value)) ? true : 'Device ID must be an integer')}
                />
            </View>

            <Button marginTop={'size-200'} type={'submit'} isPending={isApplyPending} isDisabled={isApplyDisabled}>
                Apply
            </Button>
        </form>
    );
};
