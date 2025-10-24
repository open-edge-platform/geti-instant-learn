/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect, useRef } from 'react';

import { Button, Flex } from '@geti/ui';

import { usePromptMode } from '../prompt-sidebar/prompt-modes/prompt-modes.component';
import { useWebRTCConnection } from './web-rtc/web-rtc-connection-provider';

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

const CaptureForVisualPrompt = () => {
    return (
        <Button variant={'primary'} staticColor={'white'} alignSelf={'center'} style={'fill'}>
            Capture for visual prompt
        </Button>
    );
};

export const Stream = () => {
    const videoRef = useStreamToVideo();
    const promptMode = usePromptMode();

    return (
        <Flex width={'100%'} height={'100%'} direction={'column'}>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                controls={false}
                width={'100%'}
                height={'100%'}
                style={{ flex: 1 }}
            />
            {promptMode === 'visual' && <CaptureForVisualPrompt />}
        </Flex>
    );
};
