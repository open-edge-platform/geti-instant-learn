/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, useMemo } from 'react';

import { $api, type DeviceInfoType } from '@/api';
import { setModelLoading } from '@/features/model-loading';
import { useCurrentProject, useProjectIdentifier } from '@/hooks';
import { Item, Picker } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';

import { useUpdateProject } from '../../project/api/use-update-project.hook';

const AUTO_KEY = 'auto';

const deviceKey = (device: DeviceInfoType): string => {
    if (device.type === 'cpu' || device.index === null || device.index === undefined) {
        return device.type;
    }
    return `${device.type}-${device.index}`;
};

const deviceLabel = (device: DeviceInfoType, all: DeviceInfoType[]): string => {
    let label = device.name;
    if (device.memory) {
        const gb = Math.round(device.memory / (1024 * 1024 * 1024));
        label += ` (${gb} GB)`;
    }
    const collidesWithSibling = all.some(
        (other) => other !== device && other.type === device.type && other.name === device.name
    );
    if (collidesWithSibling && device.index !== null && device.index !== undefined) {
        label += ` [${device.index}]`;
    }
    return label;
};

export const InferenceDevice = () => {
    const { projectId } = useProjectIdentifier();
    const { data: project } = useCurrentProject();
    const { mutate: updateProject } = useUpdateProject();
    const queryClient = useQueryClient();

    const { data: devices } = $api.useSuspenseQuery('get', '/api/v1/system/devices');

    const items = useMemo(
        () => [
            { key: AUTO_KEY, label: 'Auto' },
            ...devices.map((d) => ({ key: deviceKey(d), label: deviceLabel(d, devices) })),
        ],
        [devices]
    );

    const handleSelectionChange = (key: Key | null) => {
        if (key === null) {
            return;
        }
        const value = String(key);
        if (value === project.device) {
            return;
        }
        updateProject(projectId, { device: value }, () => {
            setModelLoading(queryClient, projectId);
        });
    };

    return (
        <Picker
            aria-label='inference device'
            label='Inference device'
            items={items}
            selectedKey={project.device ?? AUTO_KEY}
            onSelectionChange={handleSelectionChange}
        >
            {(item) => <Item key={item.key}>{item.label}</Item>}
        </Picker>
    );
};
