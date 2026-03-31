/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ModelType } from '@/api';
import type { SupportedPromptType } from './api/use-get-supported-models';

export type AnnotationType = 'polygon' | 'rectangle';

export const getAnnotationTypeFromPromptTypes = (
    supportedTypes: SupportedPromptType[]
): AnnotationType => {
    if (supportedTypes.includes('visual_rectangle')) {
        return 'rectangle';
    }
    return 'polygon';
};
