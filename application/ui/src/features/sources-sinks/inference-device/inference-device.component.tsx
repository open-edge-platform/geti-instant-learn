/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key } from 'react';

import { type DeviceInfoType } from '@/api';
import { setModelLoading } from '@/features/model-loading';
import { useInferenceDevices } from '@/features/sources-sinks/inference-device/api/use-inference-devices';
import { useCurrentProject, useProjectIdentifier } from '@/hooks';
import { Item, Picker } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';

import { useUpdateProject } from '../../project/api/use-update-project.hook';

const AUTO_KEY = 'auto';

type DeviceItem = {
    key: string;
    label: string;
};

const hasIndex = (device: DeviceInfoType): device is DeviceInfoType & { index: number } =>
    device.index !== null && device.index !== undefined;

const getDeviceKey = (device: DeviceInfoType): string =>
    device.type === 'cpu' || !hasIndex(device) ? device.type : `${device.type}-${device.index}`;

const deviceLabel = (device: DeviceInfoType, all: DeviceInfoType[]): string => {
    let label = device.name;
    if (device.memory) {
        const BYTES_PER_GB = 1024 ** 3;
        const gb = Math.round(device.memory / BYTES_PER_GB);
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

    const { data: devices } = useInferenceDevices();

    const items: DeviceItem[] = [
        { key: AUTO_KEY, label: 'Auto' },
        ...devices.map((d) => ({ key: getDeviceKey(d), label: deviceLabel(d, devices) })),
    ];

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
            {(item: DeviceItem) => <Item key={item.key}>{item.label}</Item>}
        </Picker>
    );
};
