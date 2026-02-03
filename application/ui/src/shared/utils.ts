/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnnotationPostType, AnnotationType, LabelType } from '@/api';
import { Point } from '@geti/smart-tools/types';
import { v4 as uuid } from 'uuid';

import { Annotation } from '../features/annotator/types';

export const convertAnnotationsToDTO = (inputAnnotations: Annotation[]): AnnotationPostType[] => {
    return inputAnnotations.map((annotation) => {
        if (annotation.shape.type === 'polygon') {
            return {
                config: {
                    type: 'polygon',
                    points: annotation.shape.points,
                },
                label_id: annotation.labels.at(0)?.id ?? '',
            };
        } else if (annotation.shape.type === 'rectangle') {
            const points: Point[] = [
                {
                    x: annotation.shape.x,
                    y: annotation.shape.y,
                },
                {
                    x: annotation.shape.x + annotation.shape.width,
                    y: annotation.shape.y + annotation.shape.height,
                },
            ];

            return {
                config: {
                    type: 'rectangle',
                    points,
                },
                label_id: annotation.labels.at(0)?.id ?? '',
            };
        }

        throw new Error(`Annotation of type: ${(annotation.shape as { type: string }).type} is not supported.`);
    });
};

export const convertAnnotationsFromDTO = (annotations: AnnotationType[], labels: LabelType[]): Annotation[] => {
    return annotations.map((annotation) => {
        if (annotation.config.type === 'polygon') {
            return {
                id: uuid(),
                shape: {
                    type: 'polygon',
                    points: annotation.config.points,
                },
                labels: labels.filter(({ id }) => id === annotation.label_id),
            };
        } else if (annotation.config.type === 'rectangle') {
            const [topLeft, bottomRight] = annotation.config.points;

            return {
                id: uuid(),
                shape: {
                    type: 'rectangle',
                    x: topLeft.x,
                    y: topLeft.y,
                    width: bottomRight.x - topLeft.x,
                    height: bottomRight.y - topLeft.y,
                },
                labels: labels.filter(({ id }) => id === annotation.label_id),
            };
        }

        throw new Error(`Annotation of type: ${(annotation.config as { type: string }).type} is not supported.`);
    });
};
