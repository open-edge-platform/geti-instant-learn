/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { WebcamConfig } from '@geti-prompt/api';
import { Button, TextField, View } from '@geti/ui';
import { isInteger } from 'lodash-es';

import { useCurrentProject } from '../../../project/hooks/use-current-project.hook';
import { useCreateWebcamSource } from '../hooks/use-create-webcam-source';
import { useUpdateWebcamSource } from '../hooks/use-update-webcam-source';

interface WebcamSourceProps {
    source: WebcamConfig | undefined;
}

export const WebcamSource = ({ source }: WebcamSourceProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(
        source?.config?.device_id?.toString() ?? '0'
    );
    const createWebcamSource = useCreateWebcamSource();
    const updateWebcamSource = useUpdateWebcamSource();
    const { data } = useCurrentProject();

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

    if (!data.active) {
        return <TextField label={'Device ID'} value={selectedDeviceId} isReadOnly />;
    }

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
