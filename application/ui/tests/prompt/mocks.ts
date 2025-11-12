/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VisualPromptItemType, WebcamSourceType } from '@geti-prompt/api';

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

export const DEVICE_ID = 0;
export const WEBCAM_SOURCE: WebcamSourceType = {
    connected: true,
    id: 'webcam-id',
    config: {
        seekable: false,
        device_id: DEVICE_ID,
        source_type: 'webcam',
    },
};
export const MOCK_PROMPT_ID = '123e4567-e89b-12d3-a456-426614174002';
export const MOCK_PROMPT: VisualPromptItemType = {
    id: MOCK_PROMPT_ID,
    annotations: [
        {
            config: {
                points: [
                    {
                        x: 0.1,
                        y: 0.1,
                    },
                    {
                        x: 0.5,
                        y: 0.1,
                    },
                    {
                        x: 0.5,
                        y: 0.5,
                    },
                ],
                type: 'polygon',
            },
            label_id: '123e4567-e89b-12d3-a456-426614174001',
        },
    ],
    frame_id: '123e4567-e89b-12d3-a456-426614174000',
    type: 'VISUAL',
    thumbnail: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ',
};
export const SECOND_PROMPT_ID = '123e4567-e89b-12d3-a456-426614174003';
export const SECOND_PROMPT: VisualPromptItemType = {
    id: SECOND_PROMPT_ID,
    annotations: [
        {
            config: {
                points: [
                    { x: 0.2, y: 0.2 },
                    { x: 0.6, y: 0.2 },
                    { x: 0.6, y: 0.6 },
                ],
                type: 'polygon',
            },
            label_id: '123e4567-e89b-12d3-a456-426614174001',
        },
    ],
    frame_id: '123e4567-e89b-12d3-a456-426614174001',
    type: 'VISUAL',
    thumbnail: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ',
};
