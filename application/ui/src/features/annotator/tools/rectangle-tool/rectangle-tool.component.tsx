/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType } from '@/api';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useVisualPrompt } from '../../../prompts/visual-prompt/visual-prompt-provider.component';
import { useAnnotationActions } from '../../providers/annotation-actions-provider.component';
import { useAnnotator } from '../../providers/annotator-provider.component';
import type { Label, Rect } from '../../types';
import { DrawingBox } from './drawing-box.component';

export const RectangleTool = () => {
    const { scale: zoom } = useZoom();
    const { roi, image } = useAnnotator();
    const { addAnnotations } = useAnnotationActions();
    const { selectedLabel } = useVisualPrompt();

    const handleComplete = (shapes: Rect[], annotationLabels: Label[]): void => {
        if (annotationLabels.length > 0) {
            addAnnotations(shapes, annotationLabels as LabelType[]);
        }
    };

    return <DrawingBox roi={roi} image={image} zoom={zoom} selectedLabel={selectedLabel} onComplete={handleComplete} />;
};
