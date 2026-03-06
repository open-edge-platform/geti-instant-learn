/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelType } from '@/api';

export type AnnotationType = 'polygon' | 'rectangle';

// TODO: replace with a real mapping once the backend exposes annotation_type on ModelSchema
export const getAnnotationTypeForModel = (_model: ModelType): AnnotationType => {
    return 'polygon';
};
