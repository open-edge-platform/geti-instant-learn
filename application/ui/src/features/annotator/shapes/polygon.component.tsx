/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SVGProps, useMemo } from 'react';

import type { Point } from '../types';

const getFormattedPoints = (points: Point[]): string => points.map(({ x, y }) => `${x},${y}`).join(' ');

interface PolygonProps {
    styles: SVGProps<SVGPolygonElement>;
    ariaLabel: string;
    points: Point[];
}

export const Polygon = ({ points, ariaLabel, styles }: PolygonProps) => {
    const formattedPoints = useMemo((): string => getFormattedPoints(points), [points]);

    return <polygon points={formattedPoints} aria-label={ariaLabel} {...styles} />;
};
