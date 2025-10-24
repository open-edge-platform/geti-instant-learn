/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-prompt/hooks';
import { View } from '@geti/ui';

import { ZoomTransform } from '../../components/zoom/zoom-transform';
import { Annotations } from './annotations/annotations.component';
import { useAnnotationActions } from './providers/annotation-actions-provider.component';
import { useSelectedAnnotations } from './providers/select-annotation-provider.component';
import { ToolManager } from './tools/tool-manager.component';
import { MediaItem } from './types';

const getImageUrl = (projectId: string, itemId: string) => {
    return `/api/v1/projects/${projectId}/dataset/items/${itemId}/binary`;
};

type AnnotatorCanvasProps = {
    mediaItem: MediaItem;
};

export const AnnotatorCanvas = ({ mediaItem }: AnnotatorCanvasProps) => {
    const { projectId } = useProjectIdentifier();
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    const size = { width: mediaItem.width, height: mediaItem.height };
    const imageUrl = mediaItem.url ?? getImageUrl(projectId, String(mediaItem.id));

    return (
        <ZoomTransform target={size}>
            <View position={'relative'} width={'100%'} height={'100%'}>
                <img src={imageUrl} alt='Collected data' />

                <Annotations annotations={orderedAnnotations} width={size.width} height={size.height} />
                <ToolManager />
            </View>
        </ZoomTransform>
    );
};
