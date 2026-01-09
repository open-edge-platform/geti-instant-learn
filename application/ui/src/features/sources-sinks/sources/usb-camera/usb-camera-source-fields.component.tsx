/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { TextField, View } from '@geti/ui';

import { validateDeviceId } from './utils';

export interface UsbCameraFieldsProps {
    selectedDeviceId: string;
    onSetSelectedDeviceId: (value: string) => void;
}

export const UsbCameraSourceFields = ({ selectedDeviceId, onSetSelectedDeviceId }: UsbCameraFieldsProps) => {
    return (
        <View>
            <TextField
                label={'Device ID'}
                value={selectedDeviceId}
                name='device-id'
                onChange={onSetSelectedDeviceId}
                validate={validateDeviceId}
                isRequired
            />
        </View>
    );
};
