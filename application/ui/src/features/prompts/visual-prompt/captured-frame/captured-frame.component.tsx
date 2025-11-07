/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense } from 'react';

import { Flex, Grid, Loading, minmax } from '@geti/ui';
import { AnnotationActionsProvider } from 'src/features/annotator/providers/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from 'src/features/annotator/providers/annotation-visibility-provider.component';
import { AnnotatorProvider } from 'src/features/annotator/providers/annotator-provider.component';
import { SelectAnnotationProvider } from 'src/features/annotator/providers/select-annotation-provider.component';
import { CanvasSettingsProvider } from 'src/features/annotator/settings/canvas-settings-provider.component';

import { CapturedFrameContent } from './captured-frame-content.component';
import { CapturedFrameFullScreen } from './captured-frame-full-screen.component';
import { CapturedFramePlaceholder } from './captured-frame-placeholder.component';
import { FullScreenModeProvider } from './full-screen-mode.component';

const CenteredWrapper = ({ children }: { children: ReactNode }) => (
    <Flex height={'100%'} gridRow={'1/-1'} alignItems={'center'} justifyContent={'center'}>
        {children}
    </Flex>
);
interface CapturedFrameAnnotatorProps {
    children: ReactNode;
    frameId: string;
}
const CapturedFrameProviders = ({ children, frameId }: CapturedFrameAnnotatorProps) => {
    return (
        <Suspense
            key={frameId}
            fallback={
                <CenteredWrapper>
                    <Loading mode={'inline'} />
                </CenteredWrapper>
            }
        >
            {/* Suspense key={frameId} ensures clean state reset between frames
                while allowing React Query cache to work properly */}
            <AnnotatorProvider frameId={frameId}>
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

export const CapturedFrame = ({ frameId }: { frameId: string | null }) => {
    return (
        <Grid
            width={'100%'}
            height={'size-6000'}
            areas={['labels', 'image', 'actions']}
            rows={[minmax('size-500', 'auto'), 'auto', 'size-500']}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
            }}
        >
            {frameId === null ? (
                <CenteredWrapper>
                    <CapturedFramePlaceholder />
                </CenteredWrapper>
            ) : (
                <CapturedFrameProviders frameId={frameId}>
                    <CapturedFrameContent />
                    <CapturedFrameFullScreen />
                </CapturedFrameProviders>
            )}
        </Grid>
    );
};
