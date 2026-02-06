/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { USBCameraConfig } from '@/api';
import { Item, Picker } from '@geti/ui';

export interface UsbCameraFieldsProps {
    selectedDeviceId: number;
    onSetSelectedDeviceId: (value: number) => void;
    availableUsbCameras: USBCameraConfig[];
}

export const UsbCameraSourceFields = ({
    selectedDeviceId,
    onSetSelectedDeviceId,
    availableUsbCameras,
}: UsbCameraFieldsProps) => {
    return (
        <Picker
            items={availableUsbCameras}
            selectedKey={String(selectedDeviceId)}
            onSelectionChange={(key) => {
                onSetSelectedDeviceId(Number(key));
            }}
            width={'80%'}
        >
            {(item) => <Item key={item.device_id}>{item.name}</Item>}
        </Picker>
    );
};
