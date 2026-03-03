/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { PointerEvent, useRef, useState } from 'react';

import { useEventListener } from '@/hooks';
import { clampBox, clampPointBetweenImage, pointsToRect } from '@geti/smart-tools/utils';

import { Rectangle } from '../../shapes/rectangle.component';
import type { Label, Point, Rect, RegionOfInterest } from '../../types';
import { DEFAULT_ANNOTATION_STYLES, isLeftButton } from '../../utils';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint, PointerType } from '../utils';
import { Crosshair } from './crosshair/crosshair.component';
import { useCrosshair } from './crosshair/use-crosshair.hook';

const CURSOR_OFFSET = '7 8';
const selectionCursor = '/icons/selection.svg';

interface DrawingBoxProps {
    onComplete: (shapes: Rect[], labels: Label[]) => void;
    roi: RegionOfInterest;
    image: ImageData;
    selectedLabel: Label | null;
    zoom: number;
}

export const DrawingBox = ({ roi, zoom, image, selectedLabel, onComplete }: DrawingBoxProps) => {
    const [startPoint, setStartPoint] = useState<Point | null>(null);
    const [boundingBox, setBoundingBox] = useState<Rect | null>(null);

    const canvasRef = useRef<SVGRectElement>(null);

    const clampPoint = clampPointBetweenImage(image);
    const crosshair = useCrosshair(canvasRef, zoom);

    const onPointerMove = (event: PointerEvent<SVGSVGElement>): void => {
        crosshair.onPointerMove(event);

        if (canvasRef.current === null) {
            return;
        }

        if (startPoint === null || !event.currentTarget.hasPointerCapture(event.pointerId)) {
            return;
        }

        const endPoint = clampPoint(getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom));

        setBoundingBox({ type: 'rectangle', ...clampBox(pointsToRect(startPoint, endPoint), roi) });
    };

    const onPointerDown = (event: PointerEvent<SVGSVGElement>): void => {
        if (startPoint !== null || canvasRef.current === null) {
            return;
        }

        const button = {
            button: event.button,
            buttons: event.buttons,
        };

        if (event.pointerType === PointerType.Touch || !isLeftButton(button)) {
            return;
        }

        const mouse = clampPoint(getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom));

        event.currentTarget.setPointerCapture(event.pointerId);

        setStartPoint(mouse);
        setBoundingBox({ type: 'rectangle', x: mouse.x, y: mouse.y, width: 0, height: 0 });
    };

    const onPointerUp = (event: PointerEvent<SVGSVGElement>): void => {
        if (event.pointerType === PointerType.Touch) {
            return;
        }

        if (boundingBox === null) {
            return;
        }

        // Don't make empty annotations
        if (boundingBox.width > 1 && boundingBox.height > 1) {
            onComplete([boundingBox], selectedLabel ? [selectedLabel] : []);
        }

        setCleanState();

        event.currentTarget.releasePointerCapture(event.pointerId);
    };

    const setCleanState = () => {
        setStartPoint(null);
        setBoundingBox(null);
    };

    useEventListener('keydown', (event: KeyboardEvent) => {
        if (event.key === 'Escape') {
            setCleanState();
        }
    });

    return (
        <SvgToolCanvas
            image={image}
            canvasRef={canvasRef}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerDown={onPointerDown}
            onPointerLeave={crosshair.onPointerLeave}
            style={{ cursor: `url(${selectionCursor}) ${CURSOR_OFFSET}, auto` }}
        >
            {boundingBox ? (
                <Rectangle
                    ariaLabel={'bounding box'}
                    rect={boundingBox}
                    styles={{ role: 'application', ...DEFAULT_ANNOTATION_STYLES }}
                />
            ) : null}
            <Crosshair location={crosshair.location} zoom={zoom} />
        </SvgToolCanvas>
    );
};
