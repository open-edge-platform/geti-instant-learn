/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@geti-prompt/test-fixtures';

import { LabelsPage } from '../labels/labels-page';
import { registerApiLabels } from '../labels/mocks';
import { initializeWebRTC } from '../prompt/initialize-webrtc';
import { USB_CAMERA_SOURCE } from '../prompt/mocks';
import { ANNOTATOR_PAGE_TIMEOUT, expectToHaveAnnotations, expectToNotHaveAnnotations } from './utils';

test(`Annotator`, async ({ network, page, context, streamPage, annotatorPage }) => {
    test.setTimeout(ANNOTATOR_PAGE_TIMEOUT);

    await initializeWebRTC({ page, context, network });

    registerApiLabels({ network, defaultLabels: [] });

    network.use(
        http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
            return response(200).json({
                sources: [USB_CAMERA_SOURCE],
                pagination: {
                    count: 1,
                    total: 1,
                    limit: 10,
                    offset: 0,
                },
            });
        }),

        http.put('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) =>
            response(200).json(USB_CAMERA_SOURCE)
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

        await expect(annotatorPage.getFullScreen().getCapturedFrame()).toBeVisible();
    });

    const annotatorPageFullScreen = annotatorPage.getFullScreen();
    const labelsPageFullsScreen = new LabelsPage(page, annotatorPageFullScreen.getScope());

    await test.step('Adds annotation', async () => {
        await expect(annotatorPageFullScreen.getProcessingImage()).toBeVisible({
            timeout: ANNOTATOR_PAGE_TIMEOUT,
        });
        await expect(annotatorPageFullScreen.getProcessingImage()).toBeHidden({
            timeout: ANNOTATOR_PAGE_TIMEOUT,
        });

        await annotatorPageFullScreen.addAnnotation();
        await labelsPageFullsScreen.addLabel('Label 1');

        await expect(labelsPageFullsScreen.getLabel('Label 1')).toBeVisible();
        await expectToHaveAnnotations({ annotatorPage: annotatorPageFullScreen });
    });

    await test.step('Hides/Shows annotations', async () => {
        await expectToHaveAnnotations({ annotatorPage: annotatorPageFullScreen });

        await annotatorPageFullScreen.hideAnnotations();

        await expectToNotHaveAnnotations({ annotatorPage: annotatorPageFullScreen });

        await annotatorPageFullScreen.showAnnotations();

        await expectToHaveAnnotations({ annotatorPage: annotatorPageFullScreen });
    });

    await test.step('Undoes/redoes annotations', async () => {
        await expectToHaveAnnotations({ annotatorPage: annotatorPageFullScreen });

        await annotatorPageFullScreen.undoAnnotation();

        await expectToNotHaveAnnotations({ annotatorPage: annotatorPageFullScreen });

        await annotatorPageFullScreen.redoAnnotation();

        await expectToHaveAnnotations({ annotatorPage: annotatorPageFullScreen });
    });

    await test.step('Plays with zoom', async () => {
        const initialZoom = await (await annotatorPageFullScreen.getZoomValue()).innerText();

        await annotatorPageFullScreen.zoomIn();
        await annotatorPageFullScreen.zoomIn();
        await annotatorPageFullScreen.zoomIn();

        await annotatorPageFullScreen.zoomOut();
        await annotatorPageFullScreen.zoomOut();
        await annotatorPageFullScreen.zoomOut();

        await expect(await annotatorPageFullScreen.getZoomValue()).toHaveText(initialZoom);

        await annotatorPageFullScreen.zoomIn();
        await annotatorPageFullScreen.zoomIn();
        await annotatorPageFullScreen.zoomIn();

        await annotatorPageFullScreen.fitToScreen();

        await expect(await annotatorPageFullScreen.getZoomValue()).toHaveText(initialZoom);
    });
});
