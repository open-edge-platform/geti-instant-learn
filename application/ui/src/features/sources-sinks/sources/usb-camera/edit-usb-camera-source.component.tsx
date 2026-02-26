/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { USBCameraConfig, USBCameraSourceType } from '@/api';
import { Flex, Form } from '@geti/ui';

import { useUpdateSource } from '../api/use-update-source';
import { EditSourceButtons } from '../edit-sources/edit-source-buttons.component';
import { useAvailableUsbCameras } from './api/use-available-usb-cameras';
import { NoUsbCameras } from './no-usb-cameras.component';
import { UsbCameraSourceFields } from './usb-camera-source-fields.component';

interface EditUsbCameraSourceContentProps {
    source: USBCameraSourceType;
    onSaved: () => void;
    availableUsbCameras: USBCameraConfig[];
}

const EditUsbCameraSourceContent = ({ source, onSaved, availableUsbCameras }: EditUsbCameraSourceContentProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<number>(source.config.device_id);
    const isActiveSource = source.active;

    const updateUsbCameraSource = useUpdateSource();
    const isButtonDisabled = selectedDeviceId == source.config.device_id || updateUsbCameraSource.isPending;

    const handleUpdateUsbCameraSource = (active: boolean) => {
        const name = availableUsbCameras.find((camera) => camera.device_id === selectedDeviceId)?.name;

        updateUsbCameraSource.mutate(
            source.id,
            {
                config: {
                    source_type: 'usb_camera',
                    device_id: selectedDeviceId,
                    seekable: false,
                    name,
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
                    availableUsbCameras={availableUsbCameras}
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

interface EditUsbCameraSourceProps {
    source: USBCameraSourceType;
    onSaved: () => void;
}

export const EditUsbCameraSource = ({ source, onSaved }: EditUsbCameraSourceProps) => {
    const { data } = useAvailableUsbCameras();

    if (data.length === 0) {
        return <NoUsbCameras />;
    }

    return <EditUsbCameraSourceContent source={source} onSaved={onSaved} availableUsbCameras={data} />;
};
