/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { USBCameraConfig } from '@/api';
import { Button, ButtonGroup, Flex, Form, View } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { useAvailableUsbCameras } from './api/use-available-usb-cameras';
import { NoUsbCameras } from './no-usb-cameras.component';
import { UsbCameraSourceFields } from './usb-camera-source-fields.component';

interface CreateUsbCameraSourceContentProps {
    onSaved: () => void;
    availableUsbCameras: USBCameraConfig[];
}

const CreateUsbCameraSourceContent = ({ onSaved, availableUsbCameras }: CreateUsbCameraSourceContentProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<number>(availableUsbCameras[0].device_id);

    const createUsbCameraSource = useCreateSource();

    const isApplyDisabled = createUsbCameraSource.isPending;

    const handleApply = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const cameraName = availableUsbCameras.find((camera) => camera.device_id === selectedDeviceId)?.name;

        createUsbCameraSource.mutate(
            {
                source_type: 'usb_camera',
                device_id: selectedDeviceId,
                seekable: false,
                name: cameraName,
            },
            onSaved
        );
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleApply}>
            <Flex direction={'column'} gap={'size-200'} marginTop={0}>
                <View marginTop={'size-100'}>
                    <UsbCameraSourceFields
                        selectedDeviceId={selectedDeviceId}
                        onSetSelectedDeviceId={setSelectedDeviceId}
                        availableUsbCameras={availableUsbCameras}
                    />
                </View>

                <ButtonGroup>
                    <Button type={'submit'} isPending={createUsbCameraSource.isPending} isDisabled={isApplyDisabled}>
                        Apply
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};

interface CreateUsbCameraSourceProps {
    onSaved: () => void;
}

export const CreateUsbCameraSource = ({ onSaved }: CreateUsbCameraSourceProps) => {
    const { data } = useAvailableUsbCameras();

    if (data.length === 0) {
        return <NoUsbCameras />;
    }

    return <CreateUsbCameraSourceContent onSaved={onSaved} availableUsbCameras={data} />;
};
