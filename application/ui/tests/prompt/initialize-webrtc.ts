/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { http } from '@/test-fixtures';
import { NetworkFixture } from '@msw/playwright';
import { BrowserContext, Page } from '@playwright/test';
import { HttpResponse } from 'msw';

import { mockRTCPeerConnectionScript } from './mocks';

const FRAME_ID = '1';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

export const initializeWebRTC = async ({
    page,
    context,
    network,
}: {
    page: Page;
    network: NetworkFixture;
    context: BrowserContext;
}) => {
    // Emulate prefers-reduced-motion to disable CSS animations
    await page.emulateMedia({ reducedMotion: 'reduce' });

    // Mock RTCPeerConnection to simulate a successful WebRTC connection
    await context.addInitScript(mockRTCPeerConnectionScript);

    network.use(
        http.get('/api/v1/projects/{project_id}', ({ response }) => {
            return response(200).json({
                id: 'project-id',
                name: 'Cool project',
                active: true,
            });
        }),

        http.post('/api/v1/projects/{project_id}/frames', ({ response }) => response(201).json({ frame_id: FRAME_ID })),
        http.get('/api/v1/projects/{project_id}/frames/{frame_id}', () => {
            const testImagePath = path.resolve(dirname, '../../src/assets/test.webp');
            const buffer = readFileSync(testImagePath);

            return new HttpResponse(buffer, {
                status: 200,
                headers: {
                    'Content-Type': 'image/jpeg',
                },
            });
        }),
        http.post('/api/v1/projects/{project_id}/offer', ({ response }) =>
            response(200).json({
                type: 'answer',
                sdp: 'some-sdp',
            })
        )
    );
};
