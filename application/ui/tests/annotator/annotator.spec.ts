/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamConfig } from '@geti-prompt/api';
import { expect, http, test } from '@geti-prompt/test-fixtures';

import { registerApiLabels } from '../labels/mocks';
import { initializeWebRTC } from '../prompt/initialize-webrtc';

const DEVICE_ID = 10;
const WEBCAM_SOURCE: WebcamConfig = {
    connected: true,
    id: 'webcam-id',
    config: {
        device_id: DEVICE_ID,
        source_type: 'webcam',
    },
};

test.use({ browserName: 'firefox' });
test('Annotator', async ({ network, page, context, streamPage }) => {
    await initializeWebRTC({ page, context, network });

    registerApiLabels({ network });

    network.use(
        http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
            return response(200).json({ sources: [WEBCAM_SOURCE] });
        }),

        http.put('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) =>
            response(200).json(WEBCAM_SOURCE)
        )
    );

    await test.step('Navigate to default project', async () => {
        await page.goto('/');
    });

    await test.step('Starts stream', async () => {
        await streamPage.startStream();

        await expect(streamPage.captureFrameButton).toBeVisible();
    });

    await test.step('Captures frame', async () => {
        await streamPage.captureFrame();

        await expect(page.getByAltText('Captured frame')).toBeVisible();
    });

    await test.step('Adds annotation & labels', async () => {
        await page.getByRole('button', { name: 'Select SAM Tool' }).click();

        await expect(page.getByText('Processing image, please wait...')).toBeVisible();
        await expect(page.getByText('Processing image, please wait...')).toBeHidden({ timeout: 20000 });

        const image = page.getByAltText('Captured frame');
        const box = await image.boundingBox();

        if (box) {
            // Position: middle horizontally, 20% from the bottom vertically
            const hoverX = box.x + box.width / 2;
            const hoverY = box.y + box.height * 0.8;

            // Hover to trigger preview
            await page.mouse.move(hoverX, hoverY);

            // Wait for preview to appear
            await expect(page.getByLabel('Segment anything preview')).toBeVisible({ timeout: 10000 });

            await page.mouse.click(hoverX, hoverY);

            // One for the annotation, and the other for the preview.
            expect(await page.getByLabel('annotation polygon').count()).toBe(2);
        }
    });
});
