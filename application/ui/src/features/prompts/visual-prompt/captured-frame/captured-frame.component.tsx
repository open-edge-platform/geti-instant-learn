/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense } from 'react';

import { Flex, Grid, Loading, minmax } from '@geti/ui';

import { AnnotationProviders } from '../../../annotator/annotation-providers/annotation-providers.component';
import { useVisualPrompt } from '../visual-prompt-provider.component';
import { CapturedFrameContent } from './captured-frame-content.component';
import { CapturedFrameFullScreen } from './captured-frame-full-screen.component';

interface CapturedFrameAnnotatorProps {
    children: ReactNode;
    frameId: string;
}

export const CapturedFrameProviders = ({ children, frameId }: CapturedFrameAnnotatorProps) => {
    const { prompt, labels } = useVisualPrompt();

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
            <AnnotationProviders frameId={frameId} initialAnnotationsDTO={prompt?.annotations || []} labels={labels}>
                {children}
            </AnnotationProviders>
        </Suspense>
    );
};

interface CapturedFrameProps {
    frameId: string;
}

export const CapturedFrame = ({ frameId }: CapturedFrameProps) => {
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
                data-testid={`captured-frame-${frameId}`}
            >
                <CapturedFrameContent />
            </Grid>
            <CapturedFrameFullScreen />
        </>
    );
};
