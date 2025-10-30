/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense } from 'react';

import { Loading, View } from '@geti/ui';

import { ZoomProvider } from '../../../../components/zoom/zoom.provider';
import { AnnotatorCanvas } from '../../../annotator/annotator-canvas';
import { AnnotationActionsProvider } from '../../../annotator/providers/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../annotator/providers/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../../../annotator/providers/annotator-provider.component';
import { SelectAnnotationProvider } from '../../../annotator/providers/select-annotation-provider.component';
import { CapturedFrameActions } from './captured-frame-actions.component';
import { Labels } from './labels-management/labels.component';

interface CapturedFrameAnnotatorProps {
    children: ReactNode;
    frameId: string;
}

const CapturedFrameProviders = ({ children, frameId }: CapturedFrameAnnotatorProps) => {
    return (
        <ZoomProvider>
            <Suspense
                fallback={
                    <View gridRow={'1/-1'} alignSelf={'center'}>
                        <Loading mode={'inline'} />
                    </View>
                }
            >
                <AnnotatorProvider frameId={frameId}>
                    <SelectAnnotationProvider>
                        <AnnotationActionsProvider>
                            <AnnotationVisibilityProvider>{children}</AnnotationVisibilityProvider>
                        </AnnotationActionsProvider>
                    </SelectAnnotationProvider>
                </AnnotatorProvider>
            </Suspense>
        </ZoomProvider>
    );
};

interface CapturedFrameContentProps {
    frameId: string;
}

export const CapturedFrameContent = ({ frameId }: CapturedFrameContentProps) => {
    return (
        <CapturedFrameProviders frameId={frameId}>
            <View gridArea={'labels'} backgroundColor={'gray-200'} paddingX={'size-100'} paddingY={'size-50'}>
                <Labels />
            </View>

            <View gridArea={'image'} backgroundColor={'gray-50'} overflow={'hidden'}>
                <AnnotatorCanvas frameId={frameId} />
            </View>
            <View gridArea={'actions'} backgroundColor={'gray-200'} padding={'size-100'}>
                <CapturedFrameActions />
            </View>
        </CapturedFrameProviders>
    );
};
