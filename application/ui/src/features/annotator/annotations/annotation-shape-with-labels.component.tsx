/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key } from 'react';

import polylabel from 'polylabel';

import { useAnnotationActions } from '../providers/annotation-actions-provider.component';
import type { Annotation, Polygon } from '../types';
import { AnnotationLabels } from './annotation-labels.component';
import { AnnotationShape } from './annotation-shape.component';

type AnnotationShapeProps = {
    annotation: Annotation;
};

export const AnnotationShapeWithLabels = ({ annotation }: AnnotationShapeProps) => {
    const { shape, labels } = annotation;
    const { updateAnnotations } = useAnnotationActions();

    const removeLabels = (labelId: Key | null) => {
        const updatedAnnotation = {
            ...annotation,
            labels: annotation.labels.filter((label) => label.id !== labelId),
        };

        updateAnnotations([updatedAnnotation]);
    };

    if (shape.type === 'rectangle') {
        return (
            <g transform={`translate(${shape.x}, ${shape.y})`}>
                <AnnotationShape annotation={{ ...annotation, shape: { ...shape, x: 0, y: 0 } }} />
                <AnnotationLabels labels={labels} onRemove={removeLabels} />
            </g>
        );
    }

    const polygonPoints = (shape as Polygon).points;
    const polygonCoords = [polygonPoints.map((point) => [point.x, point.y])];
    const [labelX, labelY] = polylabel(polygonCoords);

    return (
        <g transform={`translate(${labelX}, ${labelY})`}>
            <g transform={`translate(${-labelX}, ${-labelY})`}>
                <AnnotationShape annotation={annotation} />
            </g>
            <AnnotationLabels labels={labels} onRemove={removeLabels} />
        </g>
    );
};
