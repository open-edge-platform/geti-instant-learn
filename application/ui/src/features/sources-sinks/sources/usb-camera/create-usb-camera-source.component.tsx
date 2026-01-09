/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Flex, Form } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { UsbCameraSourceFields } from './usb-camera-source-fields.component';
import { isDeviceIdValid } from './utils';

interface CreateUsbCameraSourceProps {
    onSaved: () => void;
}

export const CreateUsbCameraSource = ({ onSaved }: CreateUsbCameraSourceProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<string>('0');
    const createUsbCameraSource = useCreateSource();

    const isApplyDisabled = !isDeviceIdValid(selectedDeviceId) || createUsbCameraSource.isPending;

    const handleApply = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        createUsbCameraSource.mutate(
            {
                source_type: 'usb_camera',
                device_id: parseInt(selectedDeviceId),
                seekable: false,
            },
            onSaved
        );
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleApply}>
            <Flex direction={'column'} gap={'size-200'} marginTop={0}>
                <UsbCameraSourceFields
                    selectedDeviceId={selectedDeviceId}
                    onSetSelectedDeviceId={setSelectedDeviceId}
                />

                <ButtonGroup>
                    <Button type={'submit'} isPending={createUsbCameraSource.isPending} isDisabled={isApplyDisabled}>
                        Apply
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
