/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense } from 'react';

import { Image } from '@geti-prompt/icons';
import { Content, Flex, Grid, Loading, View } from '@geti/ui';

import { ZoomProvider } from '../../../../components/zoom/zoom.provider';
import { AnnotatorCanvas } from '../../../annotator/annotator-canvas';
import { AnnotationActionsProvider } from '../../../annotator/providers/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../annotator/providers/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../../../annotator/providers/annotator-provider.component';
import { SelectAnnotationProvider } from '../../../annotator/providers/select-annotation-provider.component';
import { useSelectedFrame } from '../../../stream/selected-frame-provider.component';
import { CapturedFrameActions } from './captured-frame-actions.component';
import { Labels } from './labels-management/labels.component';

import styles from './captured-frame.module.scss';

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

const NoCapturedFramePlaceholder = () => {
    return (
        <View backgroundColor={'gray-300'} height={'100%'}>
            <Flex height={'100%'} width={'100%'} justifyContent={'center'} alignItems={'center'}>
                <Flex direction={'column'} gap={'size-100'} alignItems={'center'}>
                    <Image />
                    <Content UNSAFE_className={styles.noFramePlaceholder}>Capture frames for visual prompt</Content>
                </Flex>
            </Flex>
        </View>
    );
};

export const CapturedFrame = () => {
    const { selectedFrameId } = useSelectedFrame();

    if (selectedFrameId === null) {
        return <NoCapturedFramePlaceholder />;
    }

    return (
        <Grid
            width={'100%'}
            areas={['labels', 'image', 'actions']}
            rows={['size-500', 'size-6000', 'size-500']}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
            }}
        >
            <CapturedFrameProviders frameId={selectedFrameId}>
                <View gridArea={'labels'} backgroundColor={'gray-200'} paddingX={'size-100'} paddingY={'size-50'}>
                    <Labels />
                </View>

                <View gridArea={'image'} backgroundColor={'gray-50'} overflow={'hidden'}>
                    <AnnotatorCanvas frameId={selectedFrameId} />
                </View>
                <View gridArea={'actions'} backgroundColor={'gray-200'} padding={'size-100'}>
                    <CapturedFrameActions />
                </View>
            </CapturedFrameProviders>
        </Grid>
    );
};
