/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Image } from '@geti-prompt/icons';
import { Content, Flex, View } from '@geti/ui';

import { ZoomTransform } from '../../../../../../components/zoom/zoom-transform';

import styles from './captured-frame.module.scss';

interface CapturedFrameProps {
    frameId: string | null;
}

const NoCapturedFramePlaceholder = () => {
    return (
        <View backgroundColor={'gray-300'} height={'100%'}>
            <Flex height={'100%'} width={'100%'} justifyContent={'center'} alignItems={'center'}>
                <Flex direction={'column'} gap={'size-100'} alignItems={'center'}>
                    <View>
                        <Image />
                    </View>
                    <Content UNSAFE_className={styles.noFramePlaceholder}>Capture frames for visual prompt</Content>
                </Flex>
            </Flex>
        </View>
    );
};

export const CapturedFrame = ({ frameId }: CapturedFrameProps) => {
    const { projectId } = useProjectIdentifier();
    const size = {
        width: 500,
        height: 400,
    };

    if (frameId === null) {
        return <NoCapturedFramePlaceholder />;
    }

    // TODO: Add proxy so when we target /api it knows where to go (then we can remove localhost part)
    const imageUrl = `http://localhost:9100/api/v1/projects/${projectId}/frames/${frameId}`;

    return (
        <ZoomTransform target={size}>
            <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                <img
                    src={imageUrl}
                    alt={'Captured frame'}
                    style={{ height: '100%', width: '100%', objectFit: 'contain' }}
                />
            </div>
        </ZoomTransform>
    );
};
