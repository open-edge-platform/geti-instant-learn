/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SVGProps, useMemo } from 'react';

import type { Point, Polygon as PolygonInterface } from '../types';

const getFormattedPoints = (points: Point[]): string => points.map(({ x, y }) => `${x},${y}`).join(' ');

interface PolygonProps {
    polygon: PolygonInterface;
    styles: SVGProps<SVGPolygonElement>;
    ariaLabel: string;
}

export const Polygon = ({ polygon, ariaLabel, styles }: PolygonProps) => {
    const points = useMemo((): string => getFormattedPoints(polygon.points), [polygon]);

    return <polygon points={points} aria-label={ariaLabel} {...styles} />;
};
