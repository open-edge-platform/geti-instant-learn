/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Annotation as AnnotationType } from '../types';
import { AnnotationContext } from './annotation-context';
import { AnnotationShapeWithLabels } from './annotation-shape-with-labels.component';
import { AnnotationShape } from './annotation-shape.component';
import { SelectableAnnotation } from './selectable-annotation.component';

interface AnnotationProps {
    annotation: AnnotationType;
    withLabels?: boolean;
}
export const Annotation = ({ annotation, withLabels = false }: AnnotationProps) => {
    return (
        <AnnotationContext.Provider value={annotation}>
            <SelectableAnnotation>
                {withLabels ? (
                    <AnnotationShapeWithLabels annotation={annotation} />
                ) : (
                    <AnnotationShape annotation={annotation} />
                )}
            </SelectableAnnotation>
        </AnnotationContext.Provider>
    );
};
