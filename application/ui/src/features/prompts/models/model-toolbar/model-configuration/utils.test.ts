/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { MatcherModel, SoftMatcherModel } from '@/api';
import { getMockedModel, getMockedSam3Model } from '@/test-utils';
import { describe, expect, it } from 'vitest';

import { isMatcherModel, isPerDINOModel, isSam3Model, isSoftMatcherModel } from './utils';

describe('model type guards', () => {
    it('isMatcherModel returns true for matcher models', () => {
        const model = getMockedModel({
            config: { ...getMockedModel().config, model_type: 'matcher' } as MatcherModel['config'],
        });
        expect(isMatcherModel(model)).toBe(true);
        expect(isPerDINOModel(model)).toBe(false);
        expect(isSoftMatcherModel(model)).toBe(false);
        expect(isSam3Model(model)).toBe(false);
    });

    it('isPerDINOModel returns true for perdino models', () => {
        const model = getMockedModel();
        expect(isPerDINOModel(model)).toBe(true);
        expect(isMatcherModel(model)).toBe(false);
    });

    it('isSoftMatcherModel returns true for soft_matcher models', () => {
        const model = getMockedModel({
            config: { ...getMockedModel().config, model_type: 'soft_matcher' } as SoftMatcherModel['config'],
        });
        expect(isSoftMatcherModel(model)).toBe(true);
        expect(isMatcherModel(model)).toBe(false);
    });

    it('isSam3Model returns true for sam3 models', () => {
        const model = getMockedSam3Model();
        expect(isSam3Model(model)).toBe(true);
        expect(isMatcherModel(model)).toBe(false);
        expect(isPerDINOModel(model)).toBe(false);
        expect(isSoftMatcherModel(model)).toBe(false);
    });
});
