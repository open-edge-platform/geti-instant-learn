/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnnotationPostType, AnnotationType, LabelType } from '@geti-prompt/api';
import { Point, RegionOfInterest } from '@geti/smart-tools/types';
import { v4 as uuid } from 'uuid';

import { Annotation } from '../features/annotator/types';

const normalizePoints = (points: Point[], roi: { width: number; height: number }): Point[] => {
    return points.map((point) => {
        return {
            x: point.x / roi.width,
            y: point.y / roi.height,
        };
    });
};

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
            const normalizedPoints = normalizePoints(points, roi);

            return {
                config: {
                    type: 'rectangle',
                    points: normalizedPoints,
                },
                label_id: annotation.labels.at(0)?.id ?? '',
            };
        }

        throw new Error(`Annotation of type: ${(annotation.shape as { type: string }).type} is not supported.`);
    });
};

export const convertAnnotationsFromDTO = (
    annotations: AnnotationType[],
    labels: LabelType[],
    roi: Pick<RegionOfInterest, 'width' | 'height'>
): Annotation[] => {
    return annotations.map((annotation) => {
        if (annotation.config.type === 'polygon') {
            return {
                id: uuid(),
                shape: {
                    type: 'polygon',
                    points: denormalizePoints(annotation.config.points, roi),
                },
                // TODO: To figure out what should happen if prompt has a label with X which no longer exists,
                // maybe we should block removing the label which is used by any prompt or show an error message
                // when user tries to do that.
                labels: labels.filter(({ id }) => id === annotation.label_id),
            };
        } else if (annotation.config.type === 'rectangle') {
            const denormalizedPoints: Point[] = denormalizePoints(annotation.config.points, roi);

            return {
                id: uuid(),
                shape: {
                    type: 'rectangle',
                    x: denormalizedPoints[0].x,
                    y: denormalizedPoints[0].y,
                    width: denormalizedPoints[1].x - denormalizedPoints[0].x,
                    height: denormalizedPoints[1].y - denormalizedPoints[0].y,
                },
                labels: labels.filter(({ id }) => id === annotation.label_id),
            };
        }

        throw new Error(`Annotation of type: ${(annotation.config as { type: string }).type} is not supported.`);
    });
};
