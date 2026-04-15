/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Button, Flex } from '@geti/ui';
import { Play } from '@geti/ui/icons';

import { useStreamConnection } from '../mjpeg/stream-connection-provider';
import { Stream } from '../stream.component';

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
    const { status, start } = useStreamConnection();

    if (status === 'connected' || status === 'connecting') {
        return <Stream />;
    }

    if (status === 'idle') {
        return (
            <Container>
                <Button onPress={start} UNSAFE_className={styles.playButton} aria-label={'Start stream'}>
                    <Play width='128px' height='128px' />
                </Button>
            </Container>
        );
    }
};
