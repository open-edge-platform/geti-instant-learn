/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@geti-prompt/test-fixtures';

import { registerApiLabels } from '../labels/mocks';
import { initializeWebRTC } from '../prompt/initialize-webrtc';
import { WEBCAM_SOURCE } from '../prompt/mocks';
import { ANNOTATOR_PAGE_TIMEOUT, expectToHaveAnnotations, expectToNotHaveAnnotations } from './utils';

test('Annotator', async ({ network, page, context, streamPage, annotatorPage, labelsPage }) => {
    test.setTimeout(ANNOTATOR_PAGE_TIMEOUT);

    await initializeWebRTC({ page, context, network });

    registerApiLabels({ network, defaultLabels: [] });

    network.use(
        http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
            return response(200).json({
                sources: [WEBCAM_SOURCE],
                pagination: {
                    count: 1,
                    total: 1,
                    limit: 10,
                    offset: 0,
                },
            });
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
        await labelsPage.addLabel('Label 1');

        await expect(labelsPage.getLabel('Label 1')).toBeVisible();
        await expectToHaveAnnotations({ annotatorPage });
    });

    await test.step('Hides/Shows annotations', async () => {
        await expectToHaveAnnotations({ annotatorPage });

        await annotatorPage.hideAnnotations();

        await expectToNotHaveAnnotations({ annotatorPage });

        await annotatorPage.showAnnotations();

        await expectToHaveAnnotations({ annotatorPage });
    });

    await test.step('Undoes/redoes annotations', async () => {
        await expectToHaveAnnotations({ annotatorPage });

        await annotatorPage.undoAnnotation();

        await expectToNotHaveAnnotations({ annotatorPage });

        await annotatorPage.redoAnnotation();

        await expectToHaveAnnotations({ annotatorPage });
    });

    await test.step('Plays with zoom', async () => {
        const initialZoom = await (await annotatorPage.getZoomValue()).innerText();

        await annotatorPage.zoomIn();
        await annotatorPage.zoomIn();
        await annotatorPage.zoomIn();

        await annotatorPage.zoomOut();
        await annotatorPage.zoomOut();
        await annotatorPage.zoomOut();

        await expect(await annotatorPage.getZoomValue()).toHaveText(initialZoom);

        await annotatorPage.zoomIn();
        await annotatorPage.zoomIn();
        await annotatorPage.zoomIn();

        await annotatorPage.fitToScreen();

        await expect(await annotatorPage.getZoomValue()).toHaveText(initialZoom);
    });

    await test.step('Changes to fullscreen', async () => {
        await annotatorPage.openFullscreen();

        await expectToHaveAnnotations({ annotatorPage });

        // Open settings just for fun
        await annotatorPage.openSettings();
        await annotatorPage.closeSettings();

        await annotatorPage.closeFullscreen();
    });
});
