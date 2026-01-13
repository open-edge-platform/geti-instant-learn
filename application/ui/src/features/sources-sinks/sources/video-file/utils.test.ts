/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';

import { isVideoFilePathValid } from './utils';

describe('isVideoFilePathValid', () => {
    it('returns true for valid file path', () => {
        const result = isVideoFilePathValid('/path/to/video.mp4');

        expect(result).toBe(true);
    });

    it('returns false for empty string', () => {
        const result = isVideoFilePathValid('');

        expect(result).toBe(false);
    });

    it('returns false for whitespace-only string', () => {
        const result = isVideoFilePathValid('   ');

        expect(result).toBe(false);
    });

    it('returns false for string with only tabs', () => {
        const result = isVideoFilePathValid('\t\t');

        expect(result).toBe(false);
    });

    it('returns false for string with only newlines', () => {
        const result = isVideoFilePathValid('\n\n');

        expect(result).toBe(false);
    });

    it('returns false for mixed whitespace characters', () => {
        const result = isVideoFilePathValid(' \t\n ');

        expect(result).toBe(false);
    });

    it('returns true for path with leading spaces that has content after trim', () => {
        const result = isVideoFilePathValid('  /path/to/video.mp4  ');

        expect(result).toBe(true);
    });
});
