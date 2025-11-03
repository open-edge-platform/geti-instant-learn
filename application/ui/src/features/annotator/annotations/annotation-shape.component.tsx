/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Polygon } from '../shapes/polygon.component';
import { Rectangle } from '../shapes/rectangle.component';
import type { Annotation } from '../types';

type AnnotationShapeProps = {
    annotation: Annotation;
};

export const AnnotationShape = ({ annotation }: AnnotationShapeProps) => {
    const { shape, labels } = annotation;
    const color = labels.length ? labels[0].color : 'var(--annotation-fill)';

    if (shape.type === 'rectangle') {
        return (
            <Rectangle
                ariaLabel={'annotation rect'}
                x={shape.x}
                y={shape.y}
                width={shape.width}
                height={shape.height}
                styles={{
                    fill: color,
                }}
            />
        );
    }

    if (shape.type === 'polygon') {
        return (
            <Polygon
                ariaLabel={'annotation polygon'}
                points={shape.points}
                styles={{
                    fill: color,
                }}
            />
        );
    }

    return null;
};
