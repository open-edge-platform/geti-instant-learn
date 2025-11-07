/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnnotationPostType } from '@geti-prompt/api';
import { Point, RegionOfInterest } from '@geti/smart-tools/types';

import { Annotation } from '../../../annotator/types';

const normalizePoints = (points: Point[], roi: { width: number; height: number }): Point[] => {
    return points.map((point) => {
        return {
            x: point.x / roi.width,
            y: point.y / roi.height,
        };
    });
};

// It will be used in the future when getting the prompt with annotations
// eslint-disable-next-line
const denormalizePoints = (points: Point[], roi: { width: number; height: number }): Point[] => {
    return points.map((point) => {
        return {
            x: point.x * roi.width,
            y: point.y * roi.height,
        };
    });
};

export const convertAnnotationsToDTO = (
    inputAnnotations: Annotation[],
    roi: Pick<RegionOfInterest, 'width' | 'height'>
): AnnotationPostType[] => {
    return inputAnnotations.map((annotation) => {
        if (annotation.shape.type === 'polygon') {
            return {
                config: {
                    type: 'polygon',
                    points: normalizePoints(annotation.shape.points, roi),
                },
                label_id: annotation.labels.at(0)?.id,
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
            const normalizedPoints = normalizePoints(points, roi);

            return {
                config: {
                    type: 'rectangle',
                    points: normalizedPoints,
                },
            };
        }

        throw new Error(`Annotation of type: ${(annotation.shape as { type: string }).type} is not supported.`);
    });
};
