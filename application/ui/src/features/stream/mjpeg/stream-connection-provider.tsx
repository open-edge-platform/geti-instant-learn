/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, RefObject, useCallback, useContext, useEffect, useRef, useState } from 'react';

import { baseUrl } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { toast } from '@geti/ui';

import { MjpegConnection, type StreamConnectionStatus } from './mjpeg-connection';

interface StreamConnectionContextProps {
    connectionRef: RefObject<MjpegConnection | null>;
    status: StreamConnectionStatus;
    start: () => void;
    stop: () => void;
    streamUrl: string | null;
}

const StreamConnectionContext = createContext<StreamConnectionContextProps | null>(null);

const useStreamConnectionState = () => {
    const connectionRef = useRef<MjpegConnection>(null);
    const [status, setStatus] = useState<StreamConnectionStatus>('idle');
    const [streamUrl, setStreamUrl] = useState<string | null>(null);

    const { projectId } = useProjectIdentifier();

    const start = useCallback(() => {
        if (connectionRef.current) {
            const url = `${baseUrl}/api/v1/projects/${projectId}/stream`;
            setStreamUrl(url);
            connectionRef.current.start();
        }
    }, [projectId]);

    const stop = useCallback(() => {
        connectionRef.current?.stop();
        setStreamUrl(null);
    }, []);

    useEffect(() => {
        if (connectionRef.current !== null) {
            return;
        }

        const connection = new MjpegConnection();
        connectionRef.current = connection;

        const unsubscribe = connection.subscribe((event) => {
            if (event.type === 'status_change') {
                setStatus(event.status);

                if (event.status === 'failed') {
                    toast({
                        type: 'error',
                        message: 'Failed to connect to the stream. Please try restarting the connection.',
                    });
                }
            } else if (event.type === 'error') {
                console.error('Stream error:', event.error);
            }
        });

        return () => {
            unsubscribe();
            connection.stop();
            connectionRef.current = null;
        };
    }, []);

    return {
        connectionRef,
        status,
        start,
        stop,
        streamUrl,
    };
};

interface StreamConnectionProviderProps {
    children: ReactNode;
}

export const StreamConnectionProvider = ({ children }: StreamConnectionProviderProps) => {
    const state = useStreamConnectionState();

    return <StreamConnectionContext value={state}>{children}</StreamConnectionContext>;
};

export const useStreamConnection = (): StreamConnectionContextProps => {
    const context = useContext(StreamConnectionContext);

    if (context === null) {
        throw new Error('useStreamConnection must be used within a StreamConnectionProvider');
    }

    return context;
};
