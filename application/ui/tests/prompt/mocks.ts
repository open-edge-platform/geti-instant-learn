/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { USBCameraSourceType, VisualPromptItemType } from '@/api';

import { getMockedVisualPromptItem } from '../../src/test-utils/mocks/mock-prompt';

export const DEVICE_ID = 0;
export const USB_CAMERA_SOURCE: USBCameraSourceType = {
    active: true,
    id: 'usb-camera-id',
    config: {
        seekable: false,
        device_id: DEVICE_ID,
        source_type: 'usb_camera',
    },
};
export const MOCK_PROMPT_ID = '123e4567-e89b-12d3-a456-426614174002';
export const MOCK_PROMPT: VisualPromptItemType = getMockedVisualPromptItem({
    id: MOCK_PROMPT_ID,
    annotations: [
        {
            config: {
                points: [
                    {
                        x: 0.1,
                        y: 0.1,
                    },
                    {
                        x: 0.5,
                        y: 0.1,
                    },
                    {
                        x: 0.5,
                        y: 0.5,
                    },
                ],
                type: 'polygon',
            },
            label_id: '123e4567-e89b-12d3-a456-426614174001',
        },
    ],
    frame_id: '123e4567-e89b-12d3-a456-426614174000',
    thumbnail: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ',
});

const SECOND_PROMPT_ID = '123e4567-e89b-12d3-a456-426614174003';

export const SECOND_PROMPT: VisualPromptItemType = getMockedVisualPromptItem({
    id: SECOND_PROMPT_ID,
    annotations: [
        {
            config: {
                points: [
                    { x: 0.2, y: 0.2 },
                    { x: 0.6, y: 0.2 },
                    { x: 0.6, y: 0.6 },
                ],
                type: 'polygon',
            },
            label_id: '123e4567-e89b-12d3-a456-426614174001',
        },
    ],
    frame_id: '123e4567-e89b-12d3-a456-426614174001',
    thumbnail: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ',
});
