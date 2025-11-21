/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Annotation as AnnotationType } from '../types';
import { AnnotationContext } from './annotation-context';
import { AnnotationShape } from './annotation-shape.component';
import { SelectableAnnotation } from './selectable-annotation.component';

interface AnnotationProps {
    annotation: AnnotationType;
}
export const Annotation = ({ annotation }: AnnotationProps) => {
    return (
        <AnnotationContext.Provider value={annotation}>
            <SelectableAnnotation>
                <AnnotationShape annotation={annotation} />
            </SelectableAnnotation>
        </AnnotationContext.Provider>
    );
};
