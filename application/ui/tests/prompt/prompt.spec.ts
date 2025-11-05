/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamSourceType } from '@geti-prompt/api';
import { expect, http, test } from '@geti-prompt/test-fixtures';

import { LabelsPage } from '../labels/labels-page';
import { registerApiLabels } from '../labels/mocks';
import { initializeWebRTC } from './initialize-webrtc';

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

test('Prompt flow', async ({ network, page, context, streamPage, annotatorPage }) => {
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
        await streamPage.startStream();

        await expect(streamPage.captureFrameButton).toBeVisible();
    });

    await test.step('Captures frame', async () => {
        await streamPage.captureFrame();

        await expect(annotatorPage.getCapturedFrame()).toBeVisible();
    });

    await test.step('Adds annotation & labels', async () => {
        // Select bounding box tool & make annotation

        const labelsPage = new LabelsPage(page);
        const labelName = 'Label 1';

        await labelsPage.addLabel(labelName);
        await expect(labelsPage.getLabel(labelName)).toBeVisible();
    });

    await test.step('Saves prompt', async () => {
        await page.getByRole('button', { name: 'Save prompt' }).click();

        // TODO: Once the api endpoint to save prompt is integrated, complete this test
        // await... (check if the image was indeed save to the list of prompts)
    });
});
