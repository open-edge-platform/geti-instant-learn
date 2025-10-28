/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Mock RTCPeerConnection implementation for testing WebRTC connections
 * without requiring actual peer-to-peer networking.
 *
 */
export const mockRTCPeerConnectionScript = () => {
    class MockRTCPeerConnection extends EventTarget {
        connectionState = 'new';
        iceGatheringState = 'new';
        localDescription: RTCSessionDescription | null = null;
        remoteDescription: RTCSessionDescription | null = null;

        addTransceiver(_trackOrKind: string | MediaStreamTrack, _init?: RTCRtpTransceiverInit) {
            return {} as RTCRtpTransceiver;
        }

        async createOffer() {
            return { type: 'offer' as RTCSdpType, sdp: 'mock-offer' };
        }

        async setLocalDescription(desc: RTCSessionDescriptionInit) {
            this.localDescription = desc as RTCSessionDescription;
            this.iceGatheringState = 'complete';
            this.dispatchEvent(new Event('icegatheringstatechange'));
        }

        async setRemoteDescription(desc: RTCSessionDescriptionInit) {
            this.remoteDescription = desc as RTCSessionDescription;

            // Simulate connection state progression
            setTimeout(() => {
                this.connectionState = 'connecting';
                this.dispatchEvent(new Event('connectionstatechange'));

                setTimeout(() => {
                    this.connectionState = 'connected';
                    this.dispatchEvent(new Event('connectionstatechange'));
                }, 100);
            }, 100);
        }

        getTransceivers() {
            return [];
        }

        getReceivers() {
            return [];
        }

        getSenders() {
            return [];
        }

        close() {
            this.connectionState = 'closed';
        }
    }

    (window as unknown as { RTCPeerConnection: unknown }).RTCPeerConnection = MockRTCPeerConnection;
};

/**
 * Base64-encoded 1x1 red pixel JPEG image for mocking frame responses
 */
export const MOCK_FRAME_JPEG_BASE64 =
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a' +
    'HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAHwAAAQUBAQEB' +
    'AQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1Fh' +
    'ByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZ' +
    'WmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG' +
    'x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APlTWv/Z';
