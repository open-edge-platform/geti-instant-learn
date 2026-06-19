/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnnotationPostType, AnnotationType, LabelType } from '@/api';
import { v4 as uuid } from 'uuid';

import { Annotation } from '../features/annotator/types';

export const convertAnnotationsToDTO = (inputAnnotations: Annotation[]): AnnotationPostType[] => {
    return inputAnnotations.map((annotation) => ({
        config: {
            points: annotation.shape.points,
        },
        label_id: annotation.labels.at(0)?.id ?? '',
    }));
};
export const convertAnnotationsFromDTO = (annotations: AnnotationType[], labels: LabelType[]): Annotation[] => {
    return annotations.map((annotation) => ({
        id: uuid(),
        shape: {
            type: 'polygon' as const,
            points: annotation.config.points,
        },
        labels: labels.filter(({ id }) => id === annotation.label_id),
    }));
};
