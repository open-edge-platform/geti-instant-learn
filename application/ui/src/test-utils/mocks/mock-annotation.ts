/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Annotation } from '../../features/annotator/types';
import { getMockedLabel } from './mock-label';

export const getMockedAnnotation = (annotation?: Partial<Annotation>): Annotation => {
    return {
        id: 'annotation-1',
        shape: {
            type: 'polygon',
            points: [
                { x: 10, y: 20 },
                { x: 110, y: 20 },
                { x: 110, y: 70 },
                { x: 10, y: 70 },
            ],
        },
        labels: [getMockedLabel()],
        ...annotation,
    };
};
