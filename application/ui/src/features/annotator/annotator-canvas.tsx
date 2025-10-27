/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-prompt/hooks';
import { View } from '@geti/ui';

import { ZoomTransform } from '../../components/zoom/zoom-transform';
import { Annotations } from './annotations/annotations.component';
import { useAnnotationActions } from './providers/annotation-actions-provider.component';
import { useAnnotator } from './providers/annotator-provider.component';
import { useSelectedAnnotations } from './providers/select-annotation-provider.component';
import { ToolManager } from './tools/tool-manager.component';

const getImageUrl = (projectId: string, frameId: string) => {
    return `http://localhost:9100/api/v1/projects/${projectId}/frames/${frameId}`;
};

type AnnotatorCanvasProps = {
    frameId: string;
};

export const AnnotatorCanvas = ({ frameId }: AnnotatorCanvasProps) => {
    const { projectId } = useProjectIdentifier();
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { image } = useAnnotator();

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    const imageUrl = getImageUrl(projectId, frameId);

    return (
        <ZoomTransform target={image}>
            <View position={'relative'}>
                <img src={imageUrl} alt='Collected data' />

                <Annotations annotations={orderedAnnotations} width={image.width} height={image.height} />
                <ToolManager />
            </View>
        </ZoomTransform>
    );
};
