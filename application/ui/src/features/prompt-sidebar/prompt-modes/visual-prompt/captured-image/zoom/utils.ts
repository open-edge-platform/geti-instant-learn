/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ZoomState } from './types';

export const ZOOM_STEP_DIVISOR = 10;
const BUTTON_WHEEL = {
    button: 1,
    buttons: 4,
};

interface MouseButton {
    button: number;
    buttons: number;
}

const isButton = (button: MouseButton, buttonToCompare: MouseButton): boolean =>
    button.button === buttonToCompare.button || button.buttons === buttonToCompare.buttons;

export const isWheelButton = (button: MouseButton): boolean => {
    return isButton(button, BUTTON_WHEEL);
};

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
// after that this functions will be removed and used from there
export function clampBetween(min: number, value: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}
