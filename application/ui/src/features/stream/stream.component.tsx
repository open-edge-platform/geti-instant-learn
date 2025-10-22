/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useCallback, useEffect, useRef } from 'react';

import { Button, Flex, Loading } from '@geti/ui';
import { Play } from '@geti/ui/icons';

import { useWebRTCConnection } from './web-rtc/web-rtc-connection-provider';

import styles from './stream.module.scss';

const useStreamToVideo = () => {
    const videoRef = useRef<HTMLVideoElement>(null);

    const { status, webRTCConnectionRef } = useWebRTCConnection();

    const connect = useCallback(() => {
        const receivers = webRTCConnectionRef.current?.getPeerConnection()?.getReceivers();

        if (receivers === undefined) {
            return;
        }

        const stream = new MediaStream(receivers.map((receiver) => receiver.track));

        if (videoRef.current !== null && videoRef.current.srcObject !== stream) {
            videoRef.current.srcObject = stream;
        }
    }, [webRTCConnectionRef]);

    useEffect(() => {
        if (status === 'connected') {
            connect();
        }
    }, [status, connect]);

    useEffect(() => {
        const abortController = new AbortController();

        webRTCConnectionRef.current?.getPeerConnection()?.addEventListener(
            'track',
            () => {
                connect();
            },
            {
                signal: abortController.signal,
            }
        );

        return () => {
            abortController.abort();
        };
    }, [connect, webRTCConnectionRef]);

    return videoRef;
};

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

const Stream = () => {
    const videoRef = useStreamToVideo();

    return (
        <>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                controls={false}
                width={'100%'}
                height={'100%'}
                className={styles.videoStream}
            />
        </>
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
            <Container>
                <Loading mode='inline' />
            </Container>
        );
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
