/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { TextField, View } from '@geti/ui';

import { validateDeviceId } from './utils';

export interface WebcamFieldsProps {
    selectedDeviceId: string;
    onSetSelectedDeviceId: (value: string) => void;
}

export const WebcamSourceFields = ({ selectedDeviceId, onSetSelectedDeviceId }: WebcamFieldsProps) => {
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
