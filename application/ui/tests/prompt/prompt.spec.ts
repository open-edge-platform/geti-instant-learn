/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@geti-prompt/test-fixtures';
import { Page } from '@playwright/test';

import { ANNOTATOR_PAGE_TIMEOUT, expectToHaveAnnotations } from '../annotator/utils';
import { LabelsPage } from '../labels/labels-page';
import { registerApiLabels } from '../labels/mocks';
import { initializeWebRTC } from './initialize-webrtc';
import { MOCK_PROMPT, MOCK_PROMPT_ID, SECOND_PROMPT, WEBCAM_SOURCE } from './mocks';

const waitForSAM = async (page: Page) => {
    await expect(page.getByText('Processing image, please wait...')).toBeVisible({
        timeout: ANNOTATOR_PAGE_TIMEOUT,
    });
    await expect(page.getByText('Processing image, please wait...')).toBeHidden({
        timeout: ANNOTATOR_PAGE_TIMEOUT,
    });
};

test('Prompt flow', async ({ network, page, context, streamPage, annotatorPage, promptPage }) => {
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

    await test.step('Waits for SAM to load', async () => {
        await waitForSAM(page);
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
                    prompts: [MOCK_PROMPT],
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

    await test.step('Edits prompt', async () => {
        // Create a second prompt (we already have one from previous steps)
        await streamPage.captureFrame();
        await expect(annotatorPage.getCapturedFrame()).toBeVisible();
        await annotatorPage.addAnnotation();

        network.use(
            http.get('/api/v1/projects/{project_id}/prompts', ({ response }) => {
                return response(200).json({
                    prompts: [MOCK_PROMPT, SECOND_PROMPT],
                    pagination: {
                        total: 2,
                        count: 2,
                        offset: 0,
                        limit: 10,
                    },
                });
            })
        );

        await promptPage.savePrompt();
        await expect(promptPage.thumbnail).toHaveCount(2);

        // Edit the first prompt
        network.use(
            http.get('/api/v1/projects/{project_id}/prompts/{prompt_id}', ({ response }) => {
                return response(200).json(MOCK_PROMPT);
            })
        );

        await promptPage.editPrompt(MOCK_PROMPT_ID);

        await waitForSAM(page);

        // Add an annotation
        await annotatorPage.addAnnotation();

        network.use(
            http.put('/api/v1/projects/{project_id}/prompts/{prompt_id}', ({ response }) => {
                return response(200).json(MOCK_PROMPT);
            })
        );

        await promptPage.savePrompt();

        await expect(promptPage.thumbnail).toHaveCount(2);
    });

    await test.step('Deletes prompt', async () => {
        await expect(promptPage.thumbnail).toHaveCount(2);

        network.use(
            http.get('/api/v1/projects/{project_id}/prompts', ({ response }) => {
                return response(200).json({
                    prompts: [MOCK_PROMPT],
                    pagination: {
                        total: 0,
                        count: 0,
                        offset: 0,
                        limit: 10,
                    },
                });
            })
        );

        await promptPage.deletePrompt(MOCK_PROMPT_ID);

        await expect(promptPage.thumbnail).toHaveCount(1);
    });
});
