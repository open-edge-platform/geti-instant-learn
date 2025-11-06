/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamSourceType } from '@geti-prompt/api';
import { expect, http, test } from '@geti-prompt/test-fixtures';

import { registerApiLabels } from '../labels/mocks';
import { initializeWebRTC } from '../prompt/initialize-webrtc';

const DEVICE_ID = 10;
const WEBCAM_SOURCE: WebcamSourceType = {
    connected: true,
    id: 'webcam-id',
    config: {
        seekable: false,
        device_id: DEVICE_ID,
        source_type: 'webcam',
    },
};
const ANNOTATOR_PAGE_TIMEOUT = 10 * 60 * 1000;

test.use({ browserName: 'firefox' });
test('Annotator', async ({ network, page, context, streamPage, annotatorPage }) => {
    test.setTimeout(ANNOTATOR_PAGE_TIMEOUT);

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

        await expect(annotatorPage.getCapturedFrame()).toBeVisible();
    });

    await test.step('Adds annotation', async () => {
        await expect(page.getByText('Processing image, please wait...')).toBeVisible({
            timeout: ANNOTATOR_PAGE_TIMEOUT,
        });
        await expect(page.getByText('Processing image, please wait...')).toBeHidden({
            timeout: ANNOTATOR_PAGE_TIMEOUT,
        });

        await annotatorPage.addAnnotation();
    });

    await test.step('Hides/Shows annotations', async () => {
        await annotatorPage.hasAnnotations();

        await annotatorPage.hideAnnotations();

        await expect(await annotatorPage.getAnnotation()).toBeHidden();

        await annotatorPage.showAnnotations();

        await annotatorPage.hasAnnotations();
    });

    await test.step('Undoes/redoes annotations', async () => {
        await annotatorPage.hasAnnotations();

        await annotatorPage.undoAnnotation();

        await expect(await annotatorPage.getAnnotation()).toBeHidden();

        await annotatorPage.redoAnnotation();

        await annotatorPage.hasAnnotations();
    });

    await test.step('Plays with zoom', async () => {
        await annotatorPage.hasAnnotations();

        const initialZoom = await annotatorPage.getZoomValue();

        await annotatorPage.zoomIn();
        await annotatorPage.zoomIn();
        await annotatorPage.zoomIn();

        await annotatorPage.zoomOut();
        await annotatorPage.zoomOut();
        await annotatorPage.zoomOut();

        expect(await annotatorPage.getZoomValue()).toEqual(initialZoom);

        await annotatorPage.zoomIn();
        await annotatorPage.zoomIn();
        await annotatorPage.zoomIn();

        await annotatorPage.fitToScreen();

        expect(await annotatorPage.getZoomValue()).toEqual(initialZoom);
    });

    await test.step('Changes to fullscreen', async () => {
        await annotatorPage.hasAnnotations();

        await annotatorPage.openFullscreen();

        await annotatorPage.hasAnnotations();

        // Open settings just for fun
        await annotatorPage.openSettings();
        await annotatorPage.closeSettings();

        await annotatorPage.closeFullscreen();
    });
});
