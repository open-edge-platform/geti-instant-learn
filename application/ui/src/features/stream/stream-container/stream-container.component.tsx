/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { ActionButton, dimensionValue, Flex, Loading, Text } from '@geti/ui';
import { Play } from '@geti/ui/icons';

import { Stream } from '../stream.component';
import { useWebRTCConnection } from '../web-rtc/web-rtc-connection-provider';

import styles from './stream-container.module.scss';

const Container = ({ children }: { children: ReactNode }) => {
    return (
        <Flex width={'100%'} height={'100%'} alignItems={'center'} justifyContent={'center'}>
            <Flex
                height={'90%'}
                width={'90%'}
                alignItems={'center'}
                justifyContent={'center'}
                UNSAFE_className={styles.streamContainer}
            >
                {children}
            </Flex>
        </Flex>
    );
};

export const StreamContainer = () => {
    const { status, start } = useWebRTCConnection();

    if (status === 'connected') {
        return <Stream />;
    }

    if (status === 'connecting') {
        return (
            <Container>
                <Loading mode='inline' />
            </Container>
        );
    }

    if (['idle', 'disconnected', 'failed'].includes(status)) {
        return (
            <Container>
                <ActionButton onPress={start} UNSAFE_className={styles.playButton} aria-label={'Start stream'}>
                    <Play
                        color={'currentColor'}
                        width={dimensionValue('size-400')}
                        height={dimensionValue('size-400')}
                    />
                    <Text>Start stream</Text>
                </ActionButton>
            </Container>
        );
    }
};
