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
