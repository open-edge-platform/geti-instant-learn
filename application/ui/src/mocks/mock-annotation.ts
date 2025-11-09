/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Annotation } from '../features/annotator/types';
import { getMockedLabel } from './mock-label';

export const getMockedAnnotation = (annotation?: Partial<Annotation>): Annotation => {
    return {
        id: 'annotation-1',
        shape: {
            type: 'rectangle',
            x: 10,
            y: 20,
            width: 100,
            height: 50,
        },
        labels: [getMockedLabel()],
        ...annotation,
    };
};
