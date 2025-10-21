/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ZoomState } from './types';

export const ZOOM_STEP_DIVISOR = 10;

export const getZoomState =
    ({
        initialCoordinates,
        newScale,
        cursorX,
        cursorY,
    }: {
        newScale: number;
        cursorX: number;
        cursorY: number;
        initialCoordinates: ZoomState['initialCoordinates'];
    }) =>
    (prev: ZoomState) => {
        if (newScale <= initialCoordinates.scale) {
            return {
                ...prev,
                scale: initialCoordinates.scale,
                translate: { x: initialCoordinates.x, y: initialCoordinates.y },
            };
        }

        const scaleRatio = newScale / prev.scale;
        const newTranslateX = cursorX - scaleRatio * (cursorX - prev.translate.x);
        const newTranslateY = cursorY - scaleRatio * (cursorY - prev.translate.y);

        return {
            ...prev,
            scale: newScale,
            translate: { x: newTranslateX, y: newTranslateY },
        };
    };

//TODO: smart-tools/utils package will be added in separate pr
// after that declared below functions will be removed and used from there
export interface Point {
    x: number;
    y: number;
}

export interface Rect {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface RegionOfInterest {
    x: number;
    y: number;
    width: number;
    height: number;
}

export function clampBetween(min: number, value: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

export const clampBox = (
    { x, y, width, height }: RegionOfInterest,
    { x: minX, y: minY, width: maxWidth, height: maxHeight }: RegionOfInterest
): RegionOfInterest => {
    const topLeft = {
        x: Math.max(x, minX),
        y: Math.max(y, minY),
    };
    const bottomRight = {
        x: Math.max(Math.min(x + width, minX + maxWidth), minX),
        y: Math.max(Math.min(y + height, minY + maxHeight), minY),
    };
    return pointsToRect(topLeft, bottomRight);
};

export function roiFromImage(image: ImageData): RegionOfInterest {
    const { width, height } = image;

    return { x: 0, y: 0, width, height };
}

export function clampPointBetweenImage(image: ImageData): (point: Point) => Point {
    const roi = roiFromImage(image);

    return ({ x, y }: Point): Point => {
        return {
            x: clampBetween(roi.x, x, roi.x + roi.width),
            y: clampBetween(roi.y, y, roi.y + roi.height),
        };
    };
}

export function pointsToRect(startPoint: Point, endPoint: Point): RegionOfInterest {
    const topLeft = {
        x: Math.min(startPoint.x, endPoint.x),

        y: Math.min(startPoint.y, endPoint.y),
    };
    const bottomRight = {
        x: Math.max(startPoint.x, endPoint.x),
        y: Math.max(startPoint.y, endPoint.y),
    };

    return {
        x: topLeft.x,
        y: topLeft.y,
        width: bottomRight.x - topLeft.x,
        height: bottomRight.y - topLeft.y,
    };
}
