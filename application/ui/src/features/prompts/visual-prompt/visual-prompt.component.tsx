/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense } from 'react';

import { Button, Flex, Grid, Loading, minmax } from '@geti/ui';
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
                <Flex height={'100%'} gridRow={'1/-1'} alignItems={'center'} justifyContent={'center'}>
                    <Loading mode={'inline'} />
                </Flex>
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

    const savePrompt = () => {
        // TODO: PUT /api/v1/projects/{project_id}/prompts/{prompt_id}
    };

    return (
        <Flex direction={'column'} gap={'size-300'} height={'100%'}>
            <Grid
                width={'100%'}
                height={'100%'}
                maxHeight={'size-6000'}
                areas={['labels', 'image', 'actions']}
                rows={[minmax('size-500', 'auto'), 'auto', 'size-500']}
                UNSAFE_style={{
                    backgroundColor: 'var(--spectrum-global-color-gray-200)',
                }}
            >
                <CapturedFrameProviders frameId={selectedFrameId}>
                    <CapturedFrame frameId={selectedFrameId as string} />
                </CapturedFrameProviders>
            </Grid>
            <Button alignSelf={'end'} variant={'secondary'} isDisabled={selectedFrameId === null} onPress={savePrompt}>
                Save prompt
            </Button>
            <PromptThumbnailList />
        </Flex>
    );
};
