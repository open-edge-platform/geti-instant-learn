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

const FRAME_ID = '1';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

const getTestImage = () => {
    const testImagePath = path.resolve(dirname, '../../src/assets/test.webp');
    return readFileSync(testImagePath);
};

export const initializeStream = async ({
    page,
    network,
}: {
    page: Page;
    network: NetworkFixture;
    context?: BrowserContext;
}) => {
    // Emulate prefers-reduced-motion to disable CSS animations
    await page.emulateMedia({ reducedMotion: 'reduce' });

    // Intercept the MJPEG stream URL so the <img> element fires onLoad
    await page.route('**/api/v1/projects/*/stream', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'image/jpeg',
            body: getTestImage(),
        });
    });

    network.use(
        http.get('/api/v1/projects/{project_id}', ({ response }) => {
            return response(200).json({
                id: 'project-id',
                name: 'Cool project',
                active: true,
                device: 'cpu',
                prompt_mode: 'visual',
            });
        }),

        http.post('/api/v1/projects/{project_id}/frames', ({ response }) => response(201).json({ frame_id: FRAME_ID })),
        http.get('/api/v1/projects/{project_id}/frames/{frame_id}', () => {
            return new HttpResponse(getTestImage(), {
                status: 200,
                headers: {
                    'Content-Type': 'image/jpeg',
                },
            });
        })
    );
};
