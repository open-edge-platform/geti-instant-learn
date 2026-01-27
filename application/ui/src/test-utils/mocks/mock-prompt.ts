/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VisualPromptItemType, VisualPromptType } from '@/api';

export const getMockedVisualPromptItem = (
    prompt: Partial<Omit<VisualPromptItemType, 'type'>> = {}
): VisualPromptItemType => {
    return {
        type: 'VISUAL',
        thumbnail: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ',
        id: 'id',
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
        ...prompt,
    };
};

export const getMockedVisualPrompt = (prompt: Partial<Omit<VisualPromptType, 'type'>> = {}): VisualPromptType => {
    return {
        id: '123',
        frame_id: '321',
        type: 'VISUAL',
        annotations: [
            {
                label_id: '123',
                config: {
                    type: 'polygon',
                    points: [],
                },
            },
        ],
        ...prompt,
    };
};
