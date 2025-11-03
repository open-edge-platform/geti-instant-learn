/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SVGProps } from 'react';

interface RectangleProps {
    styles: SVGProps<SVGRectElement>;
    ariaLabel: string;
    x: number;
    y: number;
    width: number;
    height: number;
}
export const Rectangle = ({ x, y, width, height, ariaLabel, styles }: RectangleProps) => {
    return <rect x={x} y={y} width={width} height={height} {...styles} aria-label={ariaLabel} />;
};
