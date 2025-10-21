/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CapturedImageType } from '../../types';
import { ZoomTransform } from '../zoom/zoom-transform';
import { ToolManager } from './tools/tool-manager.component';

interface AnnotatorCanvasProps {
    image: CapturedImageType;
    size: { width: number; height: number };
}

export const AnnotatorCanvas = ({ image, size }: AnnotatorCanvasProps) => {
    return (
        <ZoomTransform target={size}>
            <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                <img
                    src={image}
                    alt={image.toString()}
                    style={{ height: '100%', width: '100%', objectFit: 'contain' }}
                />
                <ToolManager />
            </div>
        </ZoomTransform>
    );
};
