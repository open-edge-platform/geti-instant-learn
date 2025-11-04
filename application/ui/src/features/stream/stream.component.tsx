/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect, useRef } from 'react';

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Button, dimensionValue, Grid, minmax } from '@geti/ui';

import { usePromptMode } from '../prompts/prompt-modes/prompt-modes.component';
import { FramesList } from './frames-list/frames-list.component';
import { useSelectedFrame } from './selected-frame-provider.component';
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

const useCaptureFrameMutation = () => {
    return $api.useMutation('post', '/api/v1/projects/{project_id}/frames');
};

const useCaptureFrame = () => {
    const { projectId } = useProjectIdentifier();
    const captureFrameMutation = useCaptureFrameMutation();
    const { setSelectedFrameId } = useSelectedFrame();

    const captureFrame = async () => {
        captureFrameMutation.mutate(
            {
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            },
            {
                onSuccess: ({ frame_id }) => {
                    setSelectedFrameId(frame_id);
                },
            }
        );
    };

    return {
        captureFrame,
        isPending: captureFrameMutation.isPending,
    };
};

const CaptureFrameButton = () => {
    const { captureFrame, isPending } = useCaptureFrame();

    return (
        <Button
            justifySelf={'center'}
            variant={'primary'}
            staticColor={'white'}
            style={'fill'}
            onPress={captureFrame}
            isPending={isPending}
        >
            Capture
        </Button>
    );
};

const useActiveSource = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/v1/projects/{project_id}/sources', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data?.sources.find((source) => source.connected);
};

const Video = () => {
    const videoRef = useStreamToVideo();

    // eslint-disable-next-line jsx-a11y/media-has-caption
    return <video ref={videoRef} autoPlay playsInline controls={false} width={'100%'} height={'100%'} />;
};

const WebcamStream = () => {
    const promptMode = usePromptMode();

    return (
        <Grid
            height={'100%'}
            width={'100%'}
            rows={[minmax(0, '1fr'), 'max-content']}
            rowGap={'size-200'}
            UNSAFE_style={{ paddingTop: dimensionValue('size-600'), paddingBottom: dimensionValue('size-200') }}
        >
            <Video />
            {promptMode === 'visual' && <CaptureFrameButton />}
        </Grid>
    );
};

const StaticSourceStream = () => {
    const promptMode = usePromptMode();

    return (
        <Grid
            height={'100%'}
            width={'100%'}
            rows={[minmax(0, '1fr'), 'max-content', '120px']}
            rowGap={'size-200'}
            UNSAFE_style={{
                paddingTop: '48px',
            }}
        >
            <Video />
            {promptMode === 'visual' && <CaptureFrameButton />}
            <FramesList />
        </Grid>
    );
};

export const Stream = () => {
    const activeSource = useActiveSource();

    // Should never happen, just for type safety
    if (activeSource === undefined) {
        return null;
    }

    if (activeSource.config.source_type === 'webcam') {
        return <WebcamStream />;
    }

    if (activeSource.config.source_type === 'images_folder' || activeSource.config.source_type === 'video_file') {
        return <StaticSourceStream />;
    }
};
