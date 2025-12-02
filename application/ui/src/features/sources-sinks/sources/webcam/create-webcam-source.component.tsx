/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { Button } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { isDeviceIdValid } from './utils';
import { WebcamSourceFields } from './webcam-source-fields.component';

interface CreateWebcamSourceProps {
    onSaved: () => void;
}

export const CreateWebcamSource = ({ onSaved }: CreateWebcamSourceProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<string>('0');
    const createWebcamSource = useCreateSource();

    const isApplyDisabled = !isDeviceIdValid(selectedDeviceId) || createWebcamSource.isPending;

    const handleApply = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        createWebcamSource.mutate(
            {
                source_type: 'webcam',
                device_id: parseInt(selectedDeviceId),
                seekable: false,
            },
            onSaved
        );
    };

    return (
        <form onSubmit={handleApply}>
            <WebcamSourceFields selectedDeviceId={selectedDeviceId} onSetSelectedDeviceId={setSelectedDeviceId} />

            <Button
                marginTop={'size-200'}
                type={'submit'}
                isPending={createWebcamSource.isPending}
                isDisabled={isApplyDisabled}
            >
                Apply
            </Button>
        </form>
    );
};
