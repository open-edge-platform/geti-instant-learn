/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, Suspense } from 'react';

import { useCurrentProject } from '@/hooks';
import { Item, Loading, Picker } from '@geti/ui';

import { useGetDevices } from './api/use-get-devices';
import { useUpdateProjectDevice } from './api/use-update-project-device';
import { DeviceOption } from './device-selector.types';

const AUTO_DEVICE: DeviceOption = { id: 'auto', name: 'Auto' };

export const DeviceSelector = () => {
    return (
        <Suspense fallback={<Loading size={'S'} />}>
            <DeviceSelectorContent />
        </Suspense>
    );
};

const DeviceSelectorContent = () => {
    const devices = useGetDevices();
    const { data: _project } = useCurrentProject();
    const updateDevice = useUpdateProjectDevice();

    const items: DeviceOption[] = [AUTO_DEVICE, ...devices];

    // TODO: Replace with actual project device when API is ready
    const selectedDevice = 'auto';

    const handleSelectionChange = (key: Key | null): void => {
        const next = String(key);

        if (key !== null && next !== selectedDevice) {
            updateDevice.mutate(next);
        }
    };

    return (
        <Picker
            label={'Device'}
            aria-label={'device'}
            labelPosition={'side'}
            labelAlign={'end'}
            selectedKey={selectedDevice}
            onSelectionChange={handleSelectionChange}
            items={items}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
