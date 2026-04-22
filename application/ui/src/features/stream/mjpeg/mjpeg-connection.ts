/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

export type StreamConnectionStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'failed';

type StreamConnectionEvent =
    | {
          type: 'status_change';
          status: StreamConnectionStatus;
      }
    | {
          type: 'error';
          error: Error;
      };

export type Listener = (event: StreamConnectionEvent) => void;

export class MjpegConnection {
    private status: StreamConnectionStatus = 'idle';
    private listeners: Listener[] = [];

    private updateStatus(status: StreamConnectionStatus): void {
        this.status = status;
        this.emit({ type: 'status_change', status });
    }

    public getStatus(): StreamConnectionStatus {
        return this.status;
    }

    public start(): void {
        if (this.status === 'connecting' || this.status === 'connected') {
            return;
        }

        this.updateStatus('connecting');
    }

    public stop(): void {
        this.updateStatus('idle');
    }

    /** Called by the <img> element's onLoad when the first frame arrives. */
    public onFrameReceived(): void {
        if (this.status === 'connecting') {
            this.updateStatus('connected');
        }
    }

    /** Called by the <img> element's onError. */
    public onFrameError(): void {
        if (this.status === 'idle') {
            return;
        }

        this.updateStatus('failed');
        this.emit({ type: 'error', error: new Error('MJPEG stream error') });
    }

    public subscribe(listener: Listener): () => void {
        this.listeners.push(listener);

        return () => {
            this.listeners = this.listeners.filter((l) => l !== listener);
        };
    }

    private emit(event: StreamConnectionEvent): void {
        for (const listener of this.listeners) {
            listener(event);
        }
    }
}
