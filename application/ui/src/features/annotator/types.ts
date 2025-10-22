/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

export interface RegionOfInterest {
    x: number;
    y: number;
    width: number;
    height: number;
}

export type Point = {
    x: number;
    y: number;
};

export type Rect = {
    x: number;
    y: number;
    width: number;
    height: number;
};

export type Polygon = {
    points: Point[];
};

export type Shape = Rect | Polygon;

// Circle is only used for visual purposes on segment-anything tool
export type Circle = {
    readonly type: 'circle';
    readonly x: number;
    readonly y: number;
    readonly r: number;
};

export type ClipperPoint = {
    X: number;
    Y: number;
};
