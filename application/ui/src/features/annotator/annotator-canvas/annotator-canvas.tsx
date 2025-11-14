/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback } from 'react';

import { View } from '@geti/ui';
import { useHotkeys } from 'react-hotkeys-hook';

import { ZoomTransform } from '../../../components/zoom/zoom-transform';
import { HOTKEYS } from '../actions/hotkeys';
import { Annotations } from '../annotations/annotations.component';
import { useAnnotationActions } from '../providers/annotation-actions-provider.component';
import { useAnnotator } from '../providers/annotator-provider.component';
import { useSelectedAnnotations } from '../providers/select-annotation-provider.component';
import { ToolManager } from '../tools/tool-manager.component';

import styles from './annotator-canvas.module.scss';

const useDeleteAnnotationHotkey = () => {
    const { selectedAnnotations } = useSelectedAnnotations();
    const { annotations, deleteAnnotations } = useAnnotationActions();

    useHotkeys(
        HOTKEYS.deleteAnnotation,
        () => {
            const selectedIds = annotations.filter((a) => selectedAnnotations.has(a.id)).map(({ id }) => id);
            deleteAnnotations(selectedIds);
        },
        [deleteAnnotations, annotations, selectedAnnotations]
    );
};

export const AnnotatorCanvas = () => {
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { image } = useAnnotator();

    const drawImageOnCanvas = useCallback(
        (canvasRef: HTMLCanvasElement | null) => {
            if (!canvasRef) return;

            canvasRef.width = image.width;
            canvasRef.height = image.height;

            const ctx = canvasRef.getContext('2d');
            if (ctx) {
                ctx.putImageData(image, 0, 0);
            }
        },
        [image]
    );

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    useDeleteAnnotationHotkey();

    return (
        <ZoomTransform target={image}>
            <View position={'relative'}>
                <canvas aria-label='Captured frame' ref={drawImageOnCanvas} className={styles.image} />

                <Annotations annotations={orderedAnnotations} width={image.width} height={image.height} />
                <ToolManager />
            </View>
        </ZoomTransform>
    );
};
