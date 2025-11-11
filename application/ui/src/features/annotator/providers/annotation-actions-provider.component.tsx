/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext, useRef } from 'react';

import { LabelType } from '@geti-prompt/api';
import { v4 as uuid } from 'uuid';

import type { Annotation, Shape } from '../types';
import { UndoRedoProvider } from '../undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../undo-redo/use-undo-redo-state';

interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotations: (shapes: Shape[], labels: LabelType[]) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[]) => void;
    isUserReviewed: boolean;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

type AnnotationActionsProviderProps = {
    children: ReactNode;
    initialAnnotations?: Annotation[];
};
export const AnnotationActionsProvider = ({ children, initialAnnotations = [] }: AnnotationActionsProviderProps) => {
    const isDirty = useRef<boolean>(false);

    const [state, setState, undoRedoActions] = useUndoRedoState<Annotation[]>(initialAnnotations);

    const updateAnnotations = (updatedAnnotations: Annotation[]) => {
        const updatedMap = new Map(updatedAnnotations.map((ann) => [ann.id, ann]));
        setState((prevAnnotations) => prevAnnotations.map((annotation) => updatedMap.get(annotation.id) ?? annotation));
        isDirty.current = true;
    };

    const addAnnotations = (shapes: Shape[], labels: LabelType[]) => {
        setState((prevAnnotations) => [
            ...prevAnnotations,
            ...shapes.map((shape) => ({
                shape,
                id: uuid(),
                labels,
            })),
        ]);

        isDirty.current = true;
    };

    const deleteAnnotations = (annotationIds: string[]) => {
        setState((prevAnnotations) => prevAnnotations.filter((annotation) => !annotationIds.includes(annotation.id)));

        isDirty.current = true;
    };

    return (
        <AnnotationsContext.Provider
            value={{
                isUserReviewed: false,
                annotations: state,

                // Local
                addAnnotations,
                updateAnnotations,
                deleteAnnotations,
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
