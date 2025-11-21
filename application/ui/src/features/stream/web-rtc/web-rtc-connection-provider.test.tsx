/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent } from '@testing-library/react';

import { Listener, WebRTCConnectionStatus } from './web-rtc-connection';
import { useWebRTCConnection, WebRTCConnectionProvider } from './web-rtc-connection-provider';

vi.mock('./web-rtc-connection.ts', () => {
    class MockWebRTCConnection {
        status: WebRTCConnectionStatus = 'idle';
        listeners: Listener[] = [];

        public getPeerConnection() {
            return undefined;
        }

        public async start() {
            this.status = 'connected';
            this.listeners.forEach((l) => l({ type: 'status_change', status: this.status }));
        }

        public async stop() {
            this.status = 'idle';
            this.listeners.forEach((l) => l({ type: 'status_change', status: this.status }));
        }

        public subscribe(listener: Listener): () => void {
            this.listeners.push(listener);
            return () => this.unsubscribe(listener);
        }

        private unsubscribe(listener: Listener): void {
            this.listeners = this.listeners.filter((currentListener) => currentListener !== listener);
        }
    }

    return {
        WebRTCConnection: vi.fn().mockImplementation(function () {
            return new MockWebRTCConnection();
        }),
    };
});

describe('WebRTCConnectionProvider', () => {
    const App = () => {
        const { status, start, stop } = useWebRTCConnection();

        return (
            <>
                <span aria-label='status'>{status}</span>
                <button aria-label='start' onClick={start}>
                    Start
                </button>
                <button aria-label='stop' onClick={stop}>
                    Stop
                </button>
            </>
        );
    };

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('provides initial status as idle', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        expect(getByLabelText('status')).toHaveTextContent('idle');
    });

    it('updates status to connected after start', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));

        expect(getByLabelText('status')).toHaveTextContent('connected');
    });

    it('updates status to idle after stop', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));

        expect(getByLabelText('status')).toHaveTextContent('connected');

        fireEvent.click(getByLabelText('stop'));

        expect(getByLabelText('status')).toHaveTextContent('idle');
    });

    it('cleans up on unmount', async () => {
        const { unmount } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        // Access the mock constructor and spy on its instance
        const { WebRTCConnection } = await import('./web-rtc-connection');
        const mockConstructor = vi.mocked(WebRTCConnection);
        const mockInstance = mockConstructor.mock.instances[0];
        const stopSpy = vi.spyOn(mockInstance, 'stop');

        unmount();

        expect(stopSpy).toHaveBeenCalled();
    });

    it('handles status sequence: start -> stop -> start', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));
        expect(getByLabelText('status')).toHaveTextContent('connected');

        fireEvent.click(getByLabelText('stop'));
        expect(getByLabelText('status')).toHaveTextContent('idle');

        fireEvent.click(getByLabelText('start'));
        expect(getByLabelText('status')).toHaveTextContent('connected');
    });
});
