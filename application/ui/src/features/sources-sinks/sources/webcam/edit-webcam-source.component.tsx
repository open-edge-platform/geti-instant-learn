/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { WebcamSourceType } from '@geti-prompt/api';
import { Flex } from '@geti/ui';

import { useUpdateSource } from '../api/use-update-source';
import { EditSourceButtons } from '../edit-sources/edit-source-buttons.component';
import { isDeviceIdValid } from './utils';
import { WebcamSourceFields } from './webcam-source-fields.component';

interface EditWebcamSourceProps {
    source: WebcamSourceType;
    onSaved: () => void;
}

export const EditWebcamSource = ({ source, onSaved }: EditWebcamSourceProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<string>(source.config.device_id.toString());
    const isActiveSource = source.active;

    const updateWebcamSource = useUpdateSource();
    const isButtonDisabled =
        selectedDeviceId == source.config.device_id.toString() ||
        !isDeviceIdValid(selectedDeviceId) ||
        updateWebcamSource.isPending;

    const handleUpdateWebcamSource = (active: boolean) => {
        updateWebcamSource.mutate(
            source.id,
            {
                config: {
                    source_type: 'webcam',
                    device_id: parseInt(selectedDeviceId),
                    seekable: false,
                },
                active,
            },
            onSaved
        );
    };

    const handleSave = () => {
        handleUpdateWebcamSource(source.active);
    };

    const handleSaveAndConnect = () => {
        handleUpdateWebcamSource(true);
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        handleSave();
    };

    return (
        <form onSubmit={handleSubmit}>
            <Flex direction={'column'} gap={'size-200'}>
                <WebcamSourceFields selectedDeviceId={selectedDeviceId} onSetSelectedDeviceId={setSelectedDeviceId} />

                <EditSourceButtons
                    isActiveSource={isActiveSource}
                    onSave={handleSave}
                    onSaveAndConnect={handleSaveAndConnect}
                    isDisabled={isButtonDisabled}
                    isPending={updateWebcamSource.isPending}
                />
            </Flex>
        </form>
    );
};
