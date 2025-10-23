/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { useSelectedAnnotations } from '../providers/select-annotation-provider.component';
import type { Annotation, Rect } from '../types';
import { useAnnotation } from './annotation-context';

interface EditAnnotationProps {
    children: ReactNode;
}

// TODO: update this on the next PR (moving tools)
export const EditableAnnotation = ({ children }: EditAnnotationProps) => {
    const annotation = useAnnotation() as Annotation & { shape: Rect };
    const { selectedAnnotations } = useSelectedAnnotations();
    // const { scale } = useZoom();

    // const { shape } = annotation;

    const isSelected = selectedAnnotations.has(annotation.id);

    if (isSelected && selectedAnnotations.size === 1) {
        return (
            // <EditBoundingBox
            //     key={`bbox-${shape.x}-${shape.y}-${shape.width}-${shape.height}`}
            //     annotation={annotation}
            //     zoom={scale}
            // />
            null
        );
    }

    return <>{children}</>;
};
