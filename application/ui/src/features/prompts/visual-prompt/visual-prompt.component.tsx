/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense } from 'react';

import { Button, Flex, Loading, View } from '@geti/ui';
import { AnnotationActionsProvider } from 'src/features/annotator/providers/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from 'src/features/annotator/providers/annotation-visibility-provider.component';
import { AnnotatorProvider } from 'src/features/annotator/providers/annotator-provider.component';
import { SelectAnnotationProvider } from 'src/features/annotator/providers/select-annotation-provider.component';
import { CanvasSettingsProvider } from 'src/features/annotator/settings/canvas-settings-provider.component';
import { useSelectedFrame } from 'src/features/stream/selected-frame-provider.component';

import { CapturedFramePlaceholder } from './captured-frame/captured-frame-placeholder.component';
import { CapturedFrame } from './captured-frame/captured-frame.component';
import { FullScreenModeProvider } from './captured-frame/full-screen-mode.component';
import { PromptThumbnailList } from './prompt-thumbnails/prompt-thumbnail-list/prompt-thumbnail-list.component';

interface CapturedFrameAnnotatorProps {
    children: ReactNode;
    frameId: string | null;
}
const CapturedFrameProviders = ({ children, frameId }: CapturedFrameAnnotatorProps) => {
    if (frameId === null) {
        return <CapturedFramePlaceholder />;
    }

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

export const VisualPrompt = () => {
    const { selectedFrameId } = useSelectedFrame();

    return (
        <Flex height={'100%'} direction={'column'} gap={'size-300'}>
            <CapturedFrameProviders frameId={selectedFrameId}>
                <CapturedFrame frameId={selectedFrameId as string} />
            </CapturedFrameProviders>
            <Button alignSelf={'end'} variant={'secondary'} isDisabled={selectedFrameId === null}>
                Save prompt
            </Button>
            <PromptThumbnailList />
        </Flex>
    );
};
