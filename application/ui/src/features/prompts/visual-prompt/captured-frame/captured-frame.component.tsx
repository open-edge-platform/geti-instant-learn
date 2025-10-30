/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Image } from '@geti-prompt/icons';
import { Content, Flex, Grid, minmax, View } from '@geti/ui';

import { useSelectedFrame } from '../../../stream/selected-frame-provider.component';
import { CapturedFrameContent, CapturedFrameProviders } from './captured-frame-content.component';
import { CapturedFrameFullScreen } from './captured-frame-full-screen.component';

import styles from './captured-frame.module.scss';

const NoCapturedFramePlaceholder = () => {
    return (
        <View backgroundColor={'gray-300'} height={'size-6000'}>
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
            rows={[minmax('size-500', 'auto'), 'size-6000', 'size-500']}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
            }}
        >
            <CapturedFrameProviders frameId={selectedFrameId}>
                <CapturedFrameContent frameId={selectedFrameId} />
                <CapturedFrameFullScreen frameId={selectedFrameId} />
            </CapturedFrameProviders>
        </Grid>
    );
};
