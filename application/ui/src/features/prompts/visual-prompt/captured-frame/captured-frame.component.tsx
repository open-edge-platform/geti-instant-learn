/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense, useEffect, useState } from 'react';

import { Flex, Grid, Loading, minmax } from '@geti/ui';

import { CanvasSettingsProvider } from '../../../../features/annotator/actions/settings/canvas-settings-provider.component';
import { AnnotationActionsProvider } from '../../../../features/annotator/providers/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../../features/annotator/providers/annotation-visibility-provider.component';
import { AnnotatorProvider, useAnnotator } from '../../../../features/annotator/providers/annotator-provider.component';
import { SelectAnnotationProvider } from '../../../../features/annotator/providers/select-annotation-provider.component';
import { Annotation } from '../../../../features/annotator/types';
import { FullScreenModeProvider } from '../../../annotator/actions/full-screen-mode.component';
import { convertAnnotationsFromDTO } from '../api/utils';
import { useProjectLabels } from '../use-project-labels.hook';
import { useVisualPrompt } from '../visual-prompt-provider.component';
import { CapturedFrameContent } from './captured-frame-content.component';
import { CapturedFrameFullScreen } from './captured-frame-full-screen.component';

interface CapturedFrameAnnotatorProps {
    children: ReactNode;
    frameId: string;
}

const CapturedFrameAnnotationProviders = ({ children }: { children: ReactNode }) => {
    const { prompt } = useVisualPrompt();
    const labels = useProjectLabels();
    const { roi } = useAnnotator();

    const [initialAnnotations, setInitialAnnotations] = useState<Annotation[]>([]);

    useEffect(() => {
        if (!prompt || !roi) {
            setInitialAnnotations([]);
            return;
        }

        // Make sure all annotations' points are denormalized
        const converted = convertAnnotationsFromDTO(prompt.annotations, labels, roi);
        setInitialAnnotations(converted);

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [prompt?.id, roi?.width, roi?.height]);

    return (
        <SelectAnnotationProvider>
            <AnnotationActionsProvider initialAnnotations={initialAnnotations}>
                <AnnotationVisibilityProvider>
                    <FullScreenModeProvider>
                        <CanvasSettingsProvider>{children}</CanvasSettingsProvider>
                    </FullScreenModeProvider>
                </AnnotationVisibilityProvider>
            </AnnotationActionsProvider>
        </SelectAnnotationProvider>
    );
};

export const CapturedFrameProviders = ({ children, frameId }: CapturedFrameAnnotatorProps) => {
    return (
        <Suspense
            key={frameId}
            fallback={
                <Flex
                    height={'100%'}
                    alignItems={'center'}
                    justifyContent={'center'}
                    UNSAFE_style={{
                        backgroundColor: 'var(--spectrum-global-color-gray-200)',
                    }}
                >
                    <Loading mode={'inline'} />
                </Flex>
            }
        >
            {/* Suspense key={frameId} ensures clean state reset between frames
                while allowing React Query cache to work properly */}
            <AnnotatorProvider frameId={frameId}>
                <CapturedFrameAnnotationProviders>{children}</CapturedFrameAnnotationProviders>
            </AnnotatorProvider>
        </Suspense>
    );
};

export const CapturedFrame = () => {
    return (
        <>
            <Grid
                width={'100%'}
                height={'size-6000'}
                areas={['labels', 'image', 'actions']}
                rows={[minmax('size-500', 'auto'), 'auto', 'size-500']}
                UNSAFE_style={{
                    backgroundColor: 'var(--spectrum-global-color-gray-200)',
                }}
            >
                <CapturedFrameContent />
            </Grid>
            <CapturedFrameFullScreen />
        </>
    );
};
