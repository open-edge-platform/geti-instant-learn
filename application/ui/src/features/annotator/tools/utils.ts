/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type ClipperShape from '@doodle3d/clipper-js';
import Clipper from '@doodle3d/clipper-js';
import type { Shape as SmartToolsShape } from '@geti/smart-tools/types';
import { isEmpty } from 'lodash-es';

import type { ClipperPoint, Point, Polygon, RegionOfInterest, Shape } from '../types';

export enum PointerType {
    Mouse = 'mouse',
    Pen = 'pen',
    Touch = 'touch',
}

// @ts-expect-error `default` actually exists in the module
const ClipperJS = Clipper.default || Clipper;

export function convertToolShapeToGetiShape(shape: SmartToolsShape): Polygon {
    if (shape.shapeType !== 'polygon') {
        throw new Error(`Unexpected shape type from smart-tools: ${(shape as { shapeType: string }).shapeType}`);
    }
    return { type: 'polygon', points: shape.points };
}

const removeOffLimitPointsPolygon = (shape: Shape, roi: RegionOfInterest): Polygon => {
    const { width, height, x, y } = roi;
    // Build polygon "eraser" rects around ROI limits to clip off-canvas polygon parts.
    const eraserSize = 10;
    const makeRectPolygon = (rx: number, ry: number, rWidth: number, rHeight: number): Polygon => ({
        type: 'polygon',
        points: [
            { x: rx, y: ry },
            { x: rx + rWidth, y: ry },
            { x: rx + rWidth, y: ry + rHeight },
            { x: rx, y: ry + rHeight },
        ],
    });
    const topRect = makeRectPolygon(x - eraserSize, y - eraserSize, width + eraserSize * 3, eraserSize);
    const leftRect = makeRectPolygon(x - eraserSize, y - eraserSize, eraserSize, height * 2);
    const rightRect = makeRectPolygon(x + width, y - eraserSize, eraserSize, height * 2);
    const bottomRect = makeRectPolygon(x - eraserSize, y + height, width + eraserSize * 3, eraserSize);

    return [leftRect, bottomRect, rightRect, topRect].reduce(
        (accum, current) => getShapesDifference(roi, accum, current),
        shape
    ) as Polygon;
};

const convertPolygonPoints = (shape: Polygon): ClipperPoint[] => {
    return shape.points.map(({ x, y }: Point) => ({ X: x, Y: y }));
};

const transformToClipperShape = (shape: Shape): ClipperShape => {
    return new ClipperJS([convertPolygonPoints(shape)], true);
};

// ROI is geometrically a rectangle; build a clipper rect path directly (without exposing
// a `Shape` rectangle type to the rest of the app).
const roiToClipperShape = (roi: RegionOfInterest): ClipperShape => {
    const { x, y, width, height } = roi;
    const path: ClipperPoint[] = [
        { X: x, Y: y },
        { X: x + width, Y: y },
        { X: x + width, Y: y + height },
        { X: x, Y: y + height },
    ];
    return new ClipperJS([path], true);
};

const runUnionOrDifference =
    <T>(algorithm: 'union' | 'difference', formatTo: (path: ClipperPoint[]) => T) =>
    (roi: RegionOfInterest, subj: Shape, clip: Shape): T => {
        const subjShape = transformToClipperShape(subj);
        const clipShape = transformToClipperShape(clip);
        const solutionPath = subjShape[algorithm](clipShape);
        const filteredPath = filterIntersectedPathsWithRoi(roi, solutionPath);
        const biggestPath = findBiggerSubPath(filteredPath);

        return formatTo(biggestPath);
    };

const clipperShapeToPolygon = (path: ClipperPoint[]): Polygon => ({
    type: 'polygon',
    points: path.map(({ X, Y }) => ({ x: X, y: Y })),
});

const getShapesDifference = runUnionOrDifference<Polygon>('difference', clipperShapeToPolygon);

const findBiggerSubPath = (shape: ClipperShape): ClipperPoint[] => {
    const areas = shape.areas();
    const { index: shapeIndex } = areas.reduce(
        (accum: { value: number; index: number }, value, index) => {
            return value > accum.value ? { value, index } : accum;
        },
        { value: 0, index: 0 }
    );

    return shape.paths.length ? shape.paths[shapeIndex] : [];
};

const hasIntersection = (clip: ClipperShape, subj: ClipperShape) => {
    const { paths } = clip.intersect(subj);

    return !isEmpty(paths);
};

const filterIntersectedPathsWithRoi = (roi: RegionOfInterest, shape: ClipperShape): ClipperShape => {
    const newPath = shape.clone();
    const roiRect = roiToClipperShape(roi);

    newPath.paths = newPath.paths.filter((subPath) => hasIntersection(roiRect, new ClipperJS([subPath])));

    return newPath;
};

export const removeOffLimitPoints = (shape: Shape, roi: RegionOfInterest): Shape => {
    return removeOffLimitPointsPolygon(shape, roi);
};

type ElementType = SVGElement | HTMLDivElement;
export const getRelativePoint = (element: ElementType, point: Point, zoom: number): Point => {
    const rect = element.getBoundingClientRect();

    return {
        x: Math.round((point.x - rect.left) / zoom),
        y: Math.round((point.y - rect.top) / zoom),
    };
};

export const loadImage = (link: string): Promise<HTMLImageElement> =>
    new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';

        image.onload = () => resolve(image);
        image.onerror = (error) => reject(error);

        image.fetchPriority = 'high';
        image.src = link;

        if (process.env.NODE_ENV === 'test') {
            // Immediately load the media item's image
            resolve(image);
        }
    });

const drawImageOnCanvas = (img: HTMLImageElement, filter = ''): HTMLCanvasElement => {
    const canvas: HTMLCanvasElement = document.createElement('canvas');

    canvas.width = img.naturalWidth ? img.naturalWidth : img.width;
    canvas.height = img.naturalHeight ? img.naturalHeight : img.height;

    const ctx = canvas.getContext('2d');

    if (ctx) {
        const width = img.naturalWidth ? img.naturalWidth : img.width;
        const height = img.naturalHeight ? img.naturalHeight : img.height;

        ctx.filter = filter;
        ctx.drawImage(img, 0, 0, width, height);
    }

    return canvas;
};

export const getImageData = (img: HTMLImageElement): ImageData => {
    // Always return valid imageData, even if the image isn't loaded yet.
    if (img.width === 0 && img.height === 0) {
        return new ImageData(1, 1);
    }

    const canvas = drawImageOnCanvas(img);
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

    const width = img.naturalWidth ? img.naturalWidth : img.width;
    const height = img.naturalHeight ? img.naturalHeight : img.height;

    return ctx.getImageData(0, 0, width, height);
};
