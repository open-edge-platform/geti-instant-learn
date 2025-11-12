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
    type: 'rectangle';
    x: number;
    y: number;
    width: number;
    height: number;
};

export type Polygon = {
    type: 'polygon';
    points: Point[];
};

export type Shape = Rect | Polygon;

export type ClipperPoint = {
    X: number;
    Y: number;
};

export type Label = {
    id: string;
    name: string;
    color: string;
    hotkey?: string;
};

// TODO: update this once we have the final type
export type Annotation = {
    id: string;
    labels: Label[];
    shape: Shape;
};
