/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamConfig } from '@geti-prompt/api';
import { expect, http, test } from '@geti-prompt/test-fixtures';
import { HttpResponse } from 'msw';

import { MOCK_FRAME_JPEG_BASE64, mockRTCPeerConnectionScript } from './mocks';

const FRAME_ID = '1';
const DEVICE_ID = 10;
const WEBCAM_SOURCE: WebcamConfig = {
    connected: true,
    id: 'webcam-id',
    config: {
        device_id: DEVICE_ID,
        source_type: 'webcam',
    },
};

test('Prompt flow', async ({ network, page, context }) => {
    // Emulate prefers-reduced-motion to disable CSS animations
    await page.emulateMedia({ reducedMotion: 'reduce' });

    // Mock RTCPeerConnection to simulate successful WebRTC connection
    await context.addInitScript(mockRTCPeerConnectionScript);

    network.use(
        http.get('/api/v1/projects/{project_id}', ({ response }) => {
            return response(200).json({
                id: 'project-id',
                name: 'Cool project',
                active: true,
            });
        }),

        http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
            return response(200).json({ sources: [WEBCAM_SOURCE] });
        }),

        http.put('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) =>
            response(200).json(WEBCAM_SOURCE)
        ),
        http.post('/api/v1/projects/{project_id}/frames', ({ response }) => response(201).json({ frame_id: FRAME_ID })),
        http.get('/api/v1/projects/{project_id}/frames/{frame_id}', () => {
            // Return a 1x1 red pixel JPEG image (image/jpeg content type)
            const buffer = Buffer.from(MOCK_FRAME_JPEG_BASE64, 'base64');

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

    await test.step('Navigate to default project', async () => {
        await page.goto('/');
    });

    // TODO: Step to add a source. At the moment we always have a default source
    //       but we will need to enable manual addition.
    // await test.step('Add a X source', async () => {
    //     await page.getByRole('button', { name: /Input\/Output Setup/ }).click();

    //     await page.locator('input[name="device-id"]').fill('some-id');
    //     await page.getByRole('button', { name: 'Apply' }).click();

    //     // Click outside the dialog to close the sources dialog
    //     await page.click('body', { position: { x: 10, y: 10 } });
    // });

    await test.step('Starts stream', async () => {
        await page.getByRole('button', { name: 'Start stream' }).click();

        await expect(page.getByRole('button', { name: 'Capture frame' })).toBeVisible();
    });

    await test.step('Captures frame', async () => {
        await page.getByRole('button', { name: 'Capture frame' }).click();

        await expect(page.getByAltText('Captured frame')).toBeVisible();
    });

    // TODO: Complete this step once we integrate the /labels endpoints
    await test.step('Adds annotation & labels', async () => {
        // Select bounding box tool & make annotation
        // Apply label
    });

    await test.step('Saves prompt', async () => {
        await page.getByRole('button', { name: 'Save prompt' }).click();

        // TODO: Once the api endpoint to save prompt is integrated, complete this test
        // await... (check if the image was indeed save to the list of prompts)
    });
});
