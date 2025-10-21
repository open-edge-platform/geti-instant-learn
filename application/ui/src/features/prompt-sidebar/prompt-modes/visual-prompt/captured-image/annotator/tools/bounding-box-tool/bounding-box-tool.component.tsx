/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useZoom } from '../../../zoom/zoom.provider';
import { useAnnotator } from '../../annotator-provider.component';
import { DrawingBox } from '../drawing-box.component';

export const BoundingBoxTool = () => {
    const { image, size } = useAnnotator();
    //const { addAnnotations } = useAnnotationActions();
    const { scale: zoom } = useZoom();

    const addAnnotations = () => {
        //Todo: to be added
        return true;
    };

    return (
        <DrawingBox
            roi={{ x: 0, y: 0, width: size.width, height: size.height }}
            image={image}
            zoom={zoom}
            onComplete={addAnnotations}
        />
    );
};
