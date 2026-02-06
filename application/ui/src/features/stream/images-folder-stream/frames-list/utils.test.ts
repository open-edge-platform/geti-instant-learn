/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameAPIType } from '@/api';

import { fulfillWithEmptyFrames } from './utils';

describe('fulfillWithEmptyFrames', () => {
    it('returns input frames if they are empty', () => {
        const frames: FrameAPIType[] = [];

        const result = fulfillWithEmptyFrames(frames);

        expect(result).toEqual([]);
        expect(result).toBe(frames);
    });

    it('returns input frames if the index of the first frame is 0', () => {
        const frames: FrameAPIType[] = [
            { index: 0, thumbnail: 'thumb0.jpg' },
            { index: 1, thumbnail: 'thumb1.jpg' },
        ];
        const result = fulfillWithEmptyFrames(frames);

        expect(result).toEqual(frames);
        expect(result).toBe(frames);
    });

    it('returns new frames fulfilled with that are placed before the first frame', () => {
        const frames: FrameAPIType[] = [
            { index: 3, thumbnail: 'thumb3.jpg' },
            { index: 4, thumbnail: 'thumb4.jpg' },
        ];
        const result = fulfillWithEmptyFrames(frames);

        expect(result).toHaveLength(5);
        expect(result[0]).toEqual({ index: 0, thumbnail: null });
        expect(result[1]).toEqual({ index: 1, thumbnail: null });
        expect(result[2]).toEqual({ index: 2, thumbnail: null });
        expect(result[3]).toEqual({ index: 3, thumbnail: 'thumb3.jpg' });
        expect(result[4]).toEqual({ index: 4, thumbnail: 'thumb4.jpg' });
    });
});
