/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Button, Flex, Loading } from '@geti/ui';
import { Play } from '@geti/ui/icons';

import { Stream } from '../stream.component';
import { useWebRTCConnection } from '../web-rtc/web-rtc-connection-provider';

import styles from './stream-container.module.scss';

const Container = ({ children, withBackground = false }: { children: ReactNode; withBackground?: boolean }) => {
    return (
        <Flex width={'100%'} height={'100%'} alignItems={'center'} justifyContent={'center'}>
            <Flex
                height={'90%'}
                width={'90%'}
                alignItems={'center'}
                justifyContent={'center'}
                UNSAFE_className={withBackground ? styles.streamContainer : undefined}
            >
                {children}
            </Flex>
        </Flex>
    );
};

export const StreamContainer = () => {
    const { status, start } = useWebRTCConnection();

    if (status === 'connected') {
        return (
            <Container>
                <Stream />
            </Container>
        );
    }

    if (status === 'connecting') {
        return (
            <Container withBackground>
                <Loading mode='inline' />
            </Container>
        );
    }

    if (status === 'idle') {
        return (
            <Container withBackground>
                <Button onPress={start} UNSAFE_className={styles.playButton} aria-label={'Start stream'}>
                    <Play width='128px' height='128px' />
                </Button>
            </Container>
        );
    }
};
