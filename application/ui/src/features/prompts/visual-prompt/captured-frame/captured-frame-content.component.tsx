/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense } from 'react';

import { Loading, View } from '@geti/ui';

import { ZoomProvider } from '../../../../components/zoom/zoom.provider';
import { AnnotatorCanvas } from '../../../annotator/annotator-canvas';
import { AnnotatorCanvasSettings } from '../../../annotator/canvas-settings/annotator-canvas-settings.component';
import { CanvasSettingsProvider } from '../../../annotator/canvas-settings/canvas-settings-provider.component';
import { AnnotationActionsProvider } from '../../../annotator/providers/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../annotator/providers/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../../../annotator/providers/annotator-provider.component';
import { SelectAnnotationProvider } from '../../../annotator/providers/select-annotation-provider.component';
import { CapturedFrameActions } from './captured-frame-actions.component';
import { FullScreenModeProvider } from './full-screen-mode.component';
import { Labels } from './labels-management/labels.component';

interface CapturedFrameAnnotatorProps {
    children: ReactNode;
    frameId: string;
}

export const CapturedFrameProviders = ({ children, frameId }: CapturedFrameAnnotatorProps) => {
    return (
        <Suspense
            fallback={
                <View gridRow={'1/-1'} alignSelf={'center'}>
                    <Loading mode={'inline'} />
                </View>
            }
        >
            {/* key={frameId} is added here to make sure that the whole tree unmounts/mounts
                every time we capture a new frame */}
            <AnnotatorProvider frameId={frameId} key={frameId}>
                <SelectAnnotationProvider>
                    <AnnotationActionsProvider>
                        <AnnotationVisibilityProvider>
                            <FullScreenModeProvider>
                                <CanvasSettingsProvider>{children}</CanvasSettingsProvider>
                            </FullScreenModeProvider>
                        </AnnotationVisibilityProvider>
                    </AnnotationActionsProvider>
                </SelectAnnotationProvider>
            </AnnotatorProvider>
        </Suspense>
    );
};

interface CapturedFrameContentProps {
    frameId: string;
}

export const CapturedFrameContent = ({ frameId }: CapturedFrameContentProps) => {
    return (
        <ZoomProvider>
            <View gridArea={'labels'} backgroundColor={'gray-200'} paddingX={'size-100'} paddingY={'size-50'}>
                <Labels />
            </View>

            <View gridArea={'image'} backgroundColor={'gray-50'} overflow={'hidden'}>
                <AnnotatorCanvasSettings>
                    <AnnotatorCanvas frameId={frameId} />
                </AnnotatorCanvasSettings>
            </View>
            <View gridArea={'actions'} backgroundColor={'gray-200'} padding={'size-100'}>
                <CapturedFrameActions />
            </View>
        </ZoomProvider>
    );
};
