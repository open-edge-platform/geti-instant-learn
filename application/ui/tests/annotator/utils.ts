/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect } from '@/test-fixtures';

import { AnnotatorPage } from './annotator-page';

export const ANNOTATOR_PAGE_TIMEOUT = 10 * 60 * 1000;

export const expectToNotHaveAnnotations = async ({ annotatorPage }: { annotatorPage: AnnotatorPage }) => {
    await expect(annotatorPage.getAnnotation()).toHaveCount(0);
};

export const expectToHaveAnnotations = async ({ annotatorPage }: { annotatorPage: AnnotatorPage }) => {
    await expect(annotatorPage.getAnnotation()).not.toHaveCount(0, { timeout: 10000 });
};
