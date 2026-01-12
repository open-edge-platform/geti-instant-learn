/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { USBCameraSourceType } from '@geti-prompt/api';
import { Flex, Form } from '@geti/ui';

import { useUpdateSource } from '../api/use-update-source';
import { EditSourceButtons } from '../edit-sources/edit-source-buttons.component';
import { UsbCameraSourceFields } from './usb-camera-source-fields.component';
import { isDeviceIdValid } from './utils';

interface EditUsbCameraSourceProps {
    source: USBCameraSourceType;
    onSaved: () => void;
}

export const EditUsbCameraSource = ({ source, onSaved }: EditUsbCameraSourceProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<string>(source.config.device_id.toString());
    const isActiveSource = source.active;

    const updateUsbCameraSource = useUpdateSource();
    const isButtonDisabled =
        selectedDeviceId == source.config.device_id.toString() ||
        !isDeviceIdValid(selectedDeviceId) ||
        updateUsbCameraSource.isPending;

    const handleUpdateUsbCameraSource = (active: boolean) => {
        updateUsbCameraSource.mutate(
            source.id,
            {
                config: {
                    source_type: 'usb_camera',
                    device_id: parseInt(selectedDeviceId),
                    seekable: false,
                },
                active,
            },
            onSaved
        );
    };

    const handleSave = () => {
        handleUpdateUsbCameraSource(source.active);
    };

    const handleSaveAndConnect = () => {
        handleUpdateUsbCameraSource(true);
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        handleSave();
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleSubmit}>
            <Flex direction={'column'} gap={'size-200'}>
                <UsbCameraSourceFields
                    selectedDeviceId={selectedDeviceId}
                    onSetSelectedDeviceId={setSelectedDeviceId}
                />

                <EditSourceButtons
                    isActiveSource={isActiveSource}
                    onSave={handleSave}
                    onSaveAndConnect={handleSaveAndConnect}
                    isDisabled={isButtonDisabled}
                    isPending={updateUsbCameraSource.isPending}
                />
            </Flex>
        </Form>
    );
};
