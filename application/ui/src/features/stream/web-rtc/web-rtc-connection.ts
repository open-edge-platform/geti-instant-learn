/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { client } from '@/api';
import { v4 as uuid } from 'uuid';

export type WebRTCConnectionStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'failed';

type WebRTCConnectionEvent =
    | {
          type: 'status_change';
          status: WebRTCConnectionStatus;
      }
    | {
          type: 'error';
          error: Error;
      };

export type Listener = (event: WebRTCConnectionEvent) => void;

type SessionData =
    | RTCSessionDescriptionInit
    | {
          status: 'failed';
          meta: { error: 'concurrency_limit_reached'; limit: number };
      };

const CONNECTION_TIMEOUT = 5000;
const CLOSE_CONNECTION_DELAY = 500;

export class WebRTCConnection {
    private peerConnection: RTCPeerConnection | null;
    private webRTCId: string;
    private status: WebRTCConnectionStatus = 'idle';
    private listeners: Listener[] = [];
    private timeoutId?: ReturnType<typeof setTimeout>;

    constructor() {
        this.peerConnection = null;
        this.webRTCId = uuid();
    }

    public getPeerConnection(): RTCPeerConnection | null {
        return this.peerConnection;
    }

    private updateStatus(status: WebRTCConnectionStatus): void {
        this.status = status;

        this.emit({ type: 'status_change', status });
    }

    private async handleOfferResponse(data: SessionData | undefined): Promise<void> {
        if (data === undefined) {
            this.updateStatus('failed');
            return;
        }

        if ('status' in data && data.status === 'failed') {
            const errorMessage =
                data.meta.error === 'concurrency_limit_reached'
                    ? `Too many connections. Maximum limit is ${data.meta.limit}`
                    : data.meta.error;

            this.updateStatus('failed');
            this.emit({ type: 'error', error: new Error(errorMessage) });
        }

        if (this.peerConnection) {
            await this.peerConnection.setRemoteDescription(data as RTCSessionDescriptionInit);
        }
    }

    private hasActiveConnection(): boolean {
        return this.peerConnection !== null && this.status !== 'idle' && this.status !== 'disconnected';
    }

    private async fetchIceServers(): Promise<RTCIceServer[]> {
        try {
            const response = await client.GET('/api/v1/webrtc/config');

            if (response.error !== undefined) {
                console.warn('Failed to fetch WebRTC config, using defaults');
                return [];
            }

            return response.data.iceServers.map((iceServer) => ({
                urls: iceServer.urls,
                credential: iceServer.credential ?? undefined,
                username: iceServer.username ?? undefined,
            }));
        } catch (error) {
            console.warn('Error fetching WebRTC config:', error);
            return [];
        }
    }

    public async start(projectId: string): Promise<void> {
        if (this.hasActiveConnection()) {
            console.warn('WebRTC connection is already active or in progress.');
            return;
        }

        try {
            this.updateStatus('connecting');

            const iceServers = await this.fetchIceServers();

            const config: RTCConfiguration = {
                iceServers,
            };

            this.peerConnection = new RTCPeerConnection(config);

            this.timeoutId = setTimeout(() => {
                console.warn('Connection is taking longer than usual. Are you on a VPN?');
            }, CONNECTION_TIMEOUT);

            // setup peer connection
            this.peerConnection.addTransceiver('video', { direction: 'recvonly' });

            // create an offer
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);

            // wait for ice gathering
            await this.waitForIceGathering();

            const offerResponse = await this.sendOffer(projectId);

            await this.handleOfferResponse(offerResponse);

            this.setupConnectionStateListener();
        } catch (error) {
            clearTimeout(this.timeoutId);
            this.updateStatus('failed');

            this.emit({ type: 'error', error: error as Error });

            await this.stop();
        }
    }

    public async stop(): Promise<void> {
        if (this.peerConnection === null) {
            return;
        }

        const transceivers = this.peerConnection.getTransceivers();
        transceivers.forEach((transceiver) => {
            transceiver.stop();
        });

        const senders = this.peerConnection.getSenders();
        senders.forEach((sender) => {
            sender.track?.stop();
        });

        // Give a brief moment for tracks to stop before closing the connection
        await new Promise<void>((resolve) =>
            setTimeout(() => {
                if (this.peerConnection) {
                    this.peerConnection.close();
                    this.peerConnection = null;
                    this.updateStatus('idle');
                }

                resolve();
            }, CLOSE_CONNECTION_DELAY)
        );
    }

    private async sendOffer(projectId: string): Promise<SessionData | undefined> {
        if (this.peerConnection === null || this.peerConnection.localDescription === null) {
            throw new Error('Local description is not set');
        }

        const { data } = await client.POST('/api/v1/projects/{project_id}/offer', {
            body: {
                webrtc_id: this.webRTCId,
                type: this.peerConnection.localDescription.type,
                sdp: this.peerConnection.localDescription.sdp,
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });

        return data as SessionData;
    }

    private setupConnectionStateListener() {
        if (this.peerConnection === null) return;

        this.peerConnection.addEventListener('connectionstatechange', () => {
            if (this.peerConnection === null) return;

            switch (this.peerConnection.connectionState) {
                case 'connected':
                    this.updateStatus('connected');
                    clearTimeout(this.timeoutId);
                    break;
                case 'disconnected':
                    this.updateStatus('disconnected');
                    break;
                case 'failed':
                    this.updateStatus('failed');
                    this.emit({ type: 'error', error: new Error('WebRTC connection failed.') });
                    break;
                case 'closed':
                    this.updateStatus('disconnected');
                    break;
                default:
                    this.updateStatus('connecting');
                    break;
            }
        });
    }

    private async waitForIceGathering(): Promise<void> {
        await Promise.race([
            new Promise<void>((resolve) => {
                if (!this.peerConnection || this.peerConnection.iceGatheringState === 'complete') {
                    resolve();
                    return;
                }

                const checkState = () => {
                    if (this.peerConnection && this.peerConnection.iceGatheringState === 'complete') {
                        this.peerConnection.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };

                this.peerConnection?.addEventListener('icegatheringstatechange', checkState);
            }),
            new Promise<void>((_, reject) =>
                setTimeout(() => reject(new Error('ICE gathering timed out')), CONNECTION_TIMEOUT)
            ),
        ]);
    }

    public subscribe(listener: Listener): () => void {
        this.listeners.push(listener);

        return () => this.unsubscribe(listener);
    }

    private unsubscribe(listener: Listener): void {
        this.listeners = this.listeners.filter((l) => l !== listener);
    }

    private emit(event: WebRTCConnectionEvent): void {
        this.listeners.forEach((listener) => listener(event));
    }
}
