/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, RefObject, useCallback, useContext, useEffect, useRef, useState } from 'react';

import { useProjectIdentifier } from '@geti-prompt/hooks';

import { WebRTCConnection, type WebRTCConnectionStatus } from './web-rtc-connection';

interface WebRTCConnectionContextProps {
    webRTCConnectionRef: RefObject<WebRTCConnection | null>;
    status: WebRTCConnectionStatus;
    start: () => Promise<void>;
    stop?: () => void;
}

const WebRTCConnectionContext = createContext<WebRTCConnectionContextProps | null>(null);

const useWebRTCConnectionState = () => {
    const webRTCConnectionRef = useRef<WebRTCConnection>(null);
    const [status, setStatus] = useState<WebRTCConnectionStatus>('idle');

    const { projectId } = useProjectIdentifier();

    const start = useCallback(async () => {
        webRTCConnectionRef.current?.start(projectId);
    }, [projectId]);

    const stop = () => {
        webRTCConnectionRef.current?.stop();
    };

    useEffect(() => {
        if (webRTCConnectionRef.current !== null) {
            return;
        }

        const webRTCConnection = new WebRTCConnection();
        webRTCConnectionRef.current = webRTCConnection;

        const unsubscribe = webRTCConnection.subscribe((event) => {
            if (event.type === 'status_change') {
                setStatus(event.status);
            } else if (event.type === 'error') {
                console.error('WebRTC error:', event.error);
            }
        });

        return () => {
            unsubscribe();
            webRTCConnection.stop();
            webRTCConnectionRef.current = null;
        };
    }, []);

    return {
        webRTCConnectionRef,
        status,
        start,
        stop,
    };
};

interface WebRTCConnectionProviderProps {
    children: ReactNode;
}

export const WebRTCConnectionProvider = ({ children }: WebRTCConnectionProviderProps) => {
    const state = useWebRTCConnectionState();

    return <WebRTCConnectionContext value={state}>{children}</WebRTCConnectionContext>;
};

export const useWebRTCConnection = (): WebRTCConnectionContextProps => {
    const context = useContext(WebRTCConnectionContext);

    if (context === null) {
        throw new Error('useWebRTCConnection must be used within a WebRTCConnectionProvider');
    }

    return context;
};
