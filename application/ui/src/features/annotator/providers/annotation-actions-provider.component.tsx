/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext, useMemo } from 'react';

import { AnnotationType, LabelType } from '@/api';
import { v4 as uuid } from 'uuid';

import { convertAnnotationsFromDTO } from '../../../shared/utils';
import { UndoRedoProvider } from '../actions/undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../actions/undo-redo/use-undo-redo-state';
import type { Annotation, Shape } from '../types';
import { useAnnotator } from './annotator-provider.component';

interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotations: (shapes: Shape[], labels: LabelType[]) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    deleteAllAnnotations: () => void;
    deleteAnnotationByLabelId: (labelId: string) => void;
    updateAnnotations: (updatedAnnotations: Annotation[]) => void;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

type AnnotationActionsProviderProps = {
    children: ReactNode;
    initialAnnotationsDTO?: AnnotationType[];
    labels?: LabelType[];
};

export const AnnotationActionsProvider = ({
    children,
    initialAnnotationsDTO,
    labels = [],
}: AnnotationActionsProviderProps) => {
    const { roi } = useAnnotator();

    const convertedAnnotations = useMemo(() => {
        if (initialAnnotationsDTO && roi.width > 0 && roi.height > 0) {
            return convertAnnotationsFromDTO(initialAnnotationsDTO, labels);
        }

        return [];
    }, [initialAnnotationsDTO, labels, roi]);

    const [annotations, setAnnotations, undoRedoActions] = useUndoRedoState<Annotation[]>(convertedAnnotations);

    const updateAnnotations = (updatedAnnotations: Annotation[]) => {
        const updatedMap = new Map(updatedAnnotations.map((ann) => [ann.id, ann]));

        setAnnotations((prevAnnotations) =>
            prevAnnotations.map((annotation) => updatedMap.get(annotation.id) ?? annotation)
        );
    };

    const addAnnotations = (shapes: Shape[], annotationLabels: LabelType[]) => {
        const newAnnotations: Annotation[] = shapes.map((shape) => ({
            shape,
            labels: annotationLabels,
            id: uuid(),
        }));

        setAnnotations((prevAnnotations) => [...prevAnnotations, ...newAnnotations]);
    };

    const deleteAnnotations = (annotationIds: string[]) => {
        setAnnotations((prevAnnotations) =>
            prevAnnotations.filter((annotation) => !annotationIds.includes(annotation.id))
        );
    };

    const deleteAllAnnotations = () => {
        setAnnotations([]);
    };

    const deleteAnnotationByLabelId = (labelId: string) => {
        setAnnotations((prevAnnotations) =>
            prevAnnotations
                .map((annotation) => ({
                    ...annotation,
                    labels: annotation.labels.filter((label) => label.id !== labelId),
                }))
                .filter((annotation) => annotation.labels.length > 0)
        );
    };

    return (
        <AnnotationsContext.Provider
            value={{
                annotations,

                addAnnotations,
                updateAnnotations,
                deleteAnnotations,
                deleteAllAnnotations,
                deleteAnnotationByLabelId,
            }}
        >
            <UndoRedoProvider state={undoRedoActions}>{children}</UndoRedoProvider>
        </AnnotationsContext.Provider>
    );
};

export const useAnnotationActions = () => {
    const context = useContext(AnnotationsContext);

    if (context === null) {
        throw new Error('useAnnotationActions must be used within "AnnotationActionsProvider"');
    }

    return context;
};
