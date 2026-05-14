/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SoftMatcherModel } from '@/api';
import { getMockedMatcherModel, getMockedModel, getMockedSam3Model } from '@/test-utils';
import { describe, expect, it } from 'vitest';

import { isMatcherModel, isPerDINOModel, isSam3Model, isSoftMatcherModel } from './utils';

describe('model type guards', () => {
    const matcherModel = getMockedMatcherModel();
    const perDinoModel = getMockedModel();
    const softMatcherModel = getMockedModel({
        config: { ...getMockedModel().config, model_type: 'soft_matcher' } as SoftMatcherModel['config'],
    });
    const sam3Model = getMockedSam3Model();

    it('isMatcherModel returns true only for matcher models', () => {
        expect(isMatcherModel(matcherModel)).toBe(true);
        expect(isMatcherModel(perDinoModel)).toBe(false);
        expect(isMatcherModel(softMatcherModel)).toBe(false);
        expect(isMatcherModel(sam3Model)).toBe(false);
    });
    it('isPerDINOModel returns true only for perdino models', () => {
        expect(isPerDINOModel(perDinoModel)).toBe(true);
        expect(isPerDINOModel(matcherModel)).toBe(false);
        expect(isPerDINOModel(softMatcherModel)).toBe(false);
        expect(isPerDINOModel(sam3Model)).toBe(false);
    });
    it('isSoftMatcherModel returns true only for soft_matcher models', () => {
        expect(isSoftMatcherModel(softMatcherModel)).toBe(true);
        expect(isSoftMatcherModel(matcherModel)).toBe(false);
        expect(isSoftMatcherModel(perDinoModel)).toBe(false);
        expect(isSoftMatcherModel(sam3Model)).toBe(false);
    });
    it('isSam3Model returns true only for sam3 models', () => {
        expect(isSam3Model(sam3Model)).toBe(true);
        expect(isSam3Model(matcherModel)).toBe(false);
        expect(isSam3Model(perDinoModel)).toBe(false);
        expect(isSam3Model(softMatcherModel)).toBe(false);
    });
});
