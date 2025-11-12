/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamSourceType } from '@geti-prompt/api';
import { expect, http, test } from '@geti-prompt/test-fixtures';

import { ANNOTATOR_PAGE_TIMEOUT, expectToHaveAnnotations } from '../annotator/utils';
import { LabelsPage } from '../labels/labels-page';
import { registerApiLabels } from '../labels/mocks';
import { initializeWebRTC } from './initialize-webrtc';

const DEVICE_ID = 0;
const WEBCAM_SOURCE: WebcamSourceType = {
    connected: true,
    id: 'webcam-id',
    config: {
        seekable: false,
        device_id: DEVICE_ID,
        source_type: 'webcam',
    },
};
const MOCK_PROMPT_ID = '123e4567-e89b-12d3-a456-426614174002';

test('Prompt flow', async ({ network, page, context, streamPage, annotatorPage, promptPage }) => {
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

    await test.step('Waits for SAM to load', async () => {
        await expect(page.getByText('Processing image, please wait...')).toBeVisible({
            timeout: ANNOTATOR_PAGE_TIMEOUT,
        });
        await expect(page.getByText('Processing image, please wait...')).toBeHidden({
            timeout: ANNOTATOR_PAGE_TIMEOUT,
        });
    });

    await test.step('Adds a label', async () => {
        const labelsPage = new LabelsPage(page);
        const labelName = 'Label 1';

        await labelsPage.addLabel(labelName);
        await expect(labelsPage.getLabel(labelName)).toBeVisible();
    });

    await test.step('Adds an annotation', async () => {
        await expect(promptPage.savePromptButton).toBeDisabled();

        await annotatorPage.addAnnotation();

        await expectToHaveAnnotations({ annotatorPage });
        await expect(promptPage.savePromptButton).toBeEnabled();
    });

    await test.step('Saves prompt', async () => {
        network.use(
            http.get('/api/v1/projects/{project_id}/prompts', ({ response }) => {
                return response(200).json({
                    prompts: [
                        {
                            id: MOCK_PROMPT_ID,
                            annotations: [
                                {
                                    config: {
                                        points: [
                                            {
                                                x: 0.1,
                                                y: 0.1,
                                            },
                                            {
                                                x: 0.5,
                                                y: 0.1,
                                            },
                                            {
                                                x: 0.5,
                                                y: 0.5,
                                            },
                                        ],
                                        type: 'polygon',
                                    },
                                    label_id: '123e4567-e89b-12d3-a456-426614174001',
                                },
                            ],
                            frame_id: '123e4567-e89b-12d3-a456-426614174000',
                            type: 'VISUAL',
                            thumbnail: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ',
                        },
                    ],
                    pagination: {
                        total: 1,
                        count: 1,
                        offset: 0,
                        limit: 10,
                    },
                });
            })
        );

        await promptPage.savePrompt();

        await expect(promptPage.thumbnail).toHaveCount(1);
    });

    await test.step('Deletes prompt', async () => {
        await expect(promptPage.thumbnail).toHaveCount(1);

        network.use(
            http.get('/api/v1/projects/{project_id}/prompts', ({ response }) => {
                return response(200).json({
                    prompts: [],
                    pagination: {
                        total: 0,
                        count: 0,
                        offset: 0,
                        limit: 10,
                    },
                });
            })
        );

        await promptPage.deletePrompt();

        await expect(promptPage.thumbnail).toHaveCount(0);
    });
});
