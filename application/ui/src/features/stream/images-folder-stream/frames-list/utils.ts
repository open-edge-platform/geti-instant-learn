/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameAPIType } from '@geti-prompt/api';

import { type FrameType } from '../api/interface';

export const fulfillWithEmptyFrames = (frames: FrameAPIType[]): FrameType[] => {
    if (frames.length === 0) {
        return frames;
    }

    if (frames[0].index === 0) {
        return frames;
    }

    const emptyFrames: FrameType[] = [];

    for (let i = 0; i < frames[0].index; i++) {
        emptyFrames.push({
            index: i,
            thumbnail: null,
        });
    }

    return [...emptyFrames, ...frames];
};
