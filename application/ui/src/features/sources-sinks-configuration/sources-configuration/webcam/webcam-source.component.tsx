/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { WebcamSourceType } from '@geti-prompt/api';
import { Button, TextField, View } from '@geti/ui';
import { isInteger } from 'lodash-es';

import { useCurrentProject } from '../../../project/hooks/use-current-project.hook';
import { useCreateSource } from '../hooks/use-create-source';
import { useUpdateSource } from '../hooks/use-update-source';

interface WebcamSourceProps {
    source: WebcamSourceType | undefined;
}

export const WebcamSource = ({ source }: WebcamSourceProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(
        source?.config?.device_id?.toString() ?? '0'
    );
    const createWebcamSource = useCreateSource();
    const updateWebcamSource = useUpdateSource();
    const { data } = useCurrentProject();

    const isApplyPending = createWebcamSource.isPending || updateWebcamSource.isPending;

    const isApplyDisabled =
        createWebcamSource.isPending ||
        updateWebcamSource.isPending ||
        (selectedDeviceId == source?.config?.device_id && source?.connected);

    const handleApply = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (selectedDeviceId === undefined) return;

        const deviceId = parseInt(selectedDeviceId);

        if (source === undefined) {
            createWebcamSource.mutate({
                source_type: 'webcam',
                device_id: deviceId,
                seekable: false,
            });
        } else {
            updateWebcamSource.mutate(source.id, {
                source_type: 'webcam',
                device_id: deviceId,
                seekable: false,
            });
        }
    };

    if (!data.active) {
        return <TextField label={'Device ID'} value={selectedDeviceId} isReadOnly />;
    }

    return (
        <form onSubmit={handleApply}>
            <View>
                <TextField
                    label={'Device ID'}
                    value={selectedDeviceId}
                    name='device-id'
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
