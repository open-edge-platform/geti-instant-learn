/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext, useEffect, useMemo, useRef } from 'react';

import { LabelType } from '@geti-prompt/api';
import { get, isEmpty } from 'lodash-es';
import { useCurrentProject } from 'src/features/project/hooks/use-current-project.hook';
import { v4 as uuid } from 'uuid';

import type { Annotation, Shape } from '../types';
import { UndoRedoProvider } from '../undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../undo-redo/use-undo-redo-state';
import { useAnnotator } from './annotator-provider.component';

// TODO: update this type
type ServerAnnotation = Annotation;

const mapServerAnnotationsToLocal = (serverAnnotations: ServerAnnotation[]): Annotation[] => {
    return serverAnnotations.map((annotation) => {
        return {
            ...annotation,
            id: uuid(),
        } as Annotation;
    });
};

interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotations: (shapes: Shape[], labels: LabelType[]) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[]) => void;
    submitAnnotations: () => Promise<void>;
    isUserReviewed: boolean;
    isSaving: boolean;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

type AnnotationActionsProviderProps = {
    children: ReactNode;
};
export const AnnotationActionsProvider = ({ children }: AnnotationActionsProviderProps) => {
    const { frameId } = useAnnotator();

    const serverAnnotations: Annotation[] = useMemo(() => [], []);
    const fetchError = null;

    const { data: project } = useCurrentProject();

    const isDirty = useRef<boolean>(false);

    const [state, setState, undoRedoActions] = useUndoRedoState<Annotation[]>([]);

    useEffect(() => {
        if (state.length > 0) {
            setState([]);
        }

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [frameId]);

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

    const submitAnnotations = async () => {
        if (!isDirty.current) return;

        // TODO: implement saving annotations
        // const serverFormattedAnnotations = mapLocalAnnotationsToServer(localAnnotations);
        // await saveMutation.mutateAsync({
        //     params: { path: { dataset_item_id: mediaItem.id || '', project_id: projectId } },
        //     body: { annotations: serverFormattedAnnotations },
        // });

        isDirty.current = false;
    };

    useEffect(() => {
        if (!project || !serverAnnotations) return;

        const annotations = serverAnnotations ?? [];

        if (annotations.length > 0) {
            const localFormattedAnnotations = mapServerAnnotationsToLocal(annotations);

            setState(localFormattedAnnotations);

            isDirty.current = false;
        }
    }, [serverAnnotations, project, setState]);

    useEffect(() => {
        if (!isEmpty(fetchError)) {
            setState([]);
        }
    }, [fetchError, setState]);

    return (
        <AnnotationsContext.Provider
            value={{
                isUserReviewed: get(serverAnnotations, 'user_reviewed', false),
                annotations: state,

                // Local
                addAnnotations,
                updateAnnotations,
                deleteAnnotations,

                // Remote
                submitAnnotations,

                // isSaving: saveMutation.isPending,
                isSaving: false,
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
