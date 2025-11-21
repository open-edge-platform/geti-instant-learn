/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

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
    const isActiveSource = source.connected;

    const updateWebcamSource = useUpdateSource();
    const isButtonDisabled =
        selectedDeviceId == source.config.device_id.toString() ||
        !isDeviceIdValid(selectedDeviceId) ||
        updateWebcamSource.isPending;

    const handleUpdateWebcamSource = (connected: boolean) => {
        updateWebcamSource.mutate(
            source.id,
            {
                config: {
                    source_type: 'webcam',
                    device_id: parseInt(selectedDeviceId),
                    seekable: false,
                },
                connected,
            },
            onSaved
        );
    };

    const handleSave = () => {
        handleUpdateWebcamSource(false);
    };

    const handleSaveAndConnect = () => {
        handleUpdateWebcamSource(true);
    };

    return (
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
    );
};
