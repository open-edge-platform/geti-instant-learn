/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback } from 'react';

import { useStreamConnection } from './mjpeg/stream-connection-provider';

export const Video = () => {
    const { streamUrl, connectionRef } = useStreamConnection();

    const onLoad = useCallback(() => {
        connectionRef.current?.onFrameReceived();
    }, [connectionRef]);

    const onError = useCallback(() => {
        connectionRef.current?.onFrameError();
    }, [connectionRef]);

    if (streamUrl === null) {
        return null;
    }

    return (
        <img
            src={streamUrl}
            alt='Inference stream'
            onLoad={onLoad}
            onError={onError}
            width='100%'
            height='100%'
            style={{ objectFit: 'contain' }}
        />
    );
};
