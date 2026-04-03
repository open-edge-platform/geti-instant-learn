/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';

import type { SupportedPromptType } from './api/use-get-supported-models';
import { getAnnotationTypeFromPromptTypes } from './utils';

describe('getAnnotationTypeFromPromptTypes', () => {
    it('returns "rectangle" when supported types include visual_rectangle', () => {
        const types: SupportedPromptType[] = ['text', 'visual_rectangle'];
        expect(getAnnotationTypeFromPromptTypes(types)).toBe('rectangle');
    });

    it('returns "polygon" when supported types include visual_polygon only', () => {
        const types: SupportedPromptType[] = ['visual_polygon'];
        expect(getAnnotationTypeFromPromptTypes(types)).toBe('polygon');
    });

    it('returns "polygon" when supported types is empty', () => {
        expect(getAnnotationTypeFromPromptTypes([])).toBe('polygon');
    });

    it('returns "polygon" when supported types include only text', () => {
        const types: SupportedPromptType[] = ['text'];
        expect(getAnnotationTypeFromPromptTypes(types)).toBe('polygon');
    });

    it('returns "rectangle" when both visual_polygon and visual_rectangle are present', () => {
        const types: SupportedPromptType[] = ['visual_polygon', 'visual_rectangle'];
        expect(getAnnotationTypeFromPromptTypes(types)).toBe('rectangle');
    });
});
