/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';

import { LabelType } from '@geti-prompt/api';

import { useAnnotationActions } from '../../../../annotator/providers/annotation-actions-provider.component';
import { useAnnotator } from '../../../../annotator/providers/annotator-provider.component';
import type { Annotation } from '../../../../annotator/types';

import classes from './label-thumbnail.module.scss';

interface LabelThumbnailProps {
    label: LabelType;
}

const THUMBNAIL_SIZE = 32;

const getAnnotationBounds = (annotation: Annotation) => {
    const { shape } = annotation;

    if (shape.type === 'rectangle') {
        return {
            minX: shape.x,
            minY: shape.y,
            maxX: shape.x + shape.width,
            maxY: shape.y + shape.height,
            width: shape.width,
            height: shape.height,
        };
    }

    if (shape.type === 'polygon') {
        const xs = shape.points.map((p) => p.x);
        const ys = shape.points.map((p) => p.y);
        const minX = Math.min(...xs);
        const minY = Math.min(...ys);
        const maxX = Math.max(...xs);
        const maxY = Math.max(...ys);

        return {
            minX,
            minY,
            maxX,
            maxY,
            width: maxX - minX,
            height: maxY - minY,
        };
    }

    return null;
};

export const LabelThumbnail = ({ label }: LabelThumbnailProps) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { annotations } = useAnnotationActions();
    const { image } = useAnnotator();

    // First annotation with this label. Ideally we would have only 1 annotation per label
    const annotation = annotations.find((a) => a.labels.some((l) => l.id === label.id));

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !annotation) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const bounds = getAnnotationBounds(annotation);
        if (!bounds) return;

        ctx.fillStyle = '#f5f5f5';
        ctx.fillRect(0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE);

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = image.width;
        tempCanvas.height = image.height;
        const tempCtx = tempCanvas.getContext('2d');

        if (!tempCtx) return;

        // Put the full image data on the temp canvas
        tempCtx.putImageData(image, 0, 0);

        // Calculate scale to fit annotation in thumbnail with padding
        const padding = 4;
        const availableSize = THUMBNAIL_SIZE - padding * 2;
        const scaleX = availableSize / bounds.width;
        const scaleY = availableSize / bounds.height;
        const scale = Math.min(scaleX, scaleY);

        // Calculate dimensions for the scaled region
        const scaledWidth = bounds.width * scale;
        const scaledHeight = bounds.height * scale;

        // Center the thumbnail
        const offsetX = (THUMBNAIL_SIZE - scaledWidth) / 2;
        const offsetY = (THUMBNAIL_SIZE - scaledHeight) / 2;

        ctx.drawImage(
            tempCanvas,
            bounds.minX,
            bounds.minY,
            bounds.width,
            bounds.height,
            offsetX,
            offsetY,
            scaledWidth,
            scaledHeight
        );
    }, [annotation, image]);

    return <canvas ref={canvasRef} width={THUMBNAIL_SIZE} height={THUMBNAIL_SIZE} className={classes.thumbnail} />;
};
