/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@geti-prompt/test-fixtures';
import { Locator } from '@playwright/test';

import { getMockedVisualPromptItem } from '../../src/test-utils/mocks/mock-prompt';
import { PromptPage } from '../annotator/prompt-page';
import { ANNOTATOR_PAGE_TIMEOUT, expectToHaveAnnotations } from '../annotator/utils';
import { LabelsPage } from '../labels/labels-page';
import { registerApiLabels } from '../labels/mocks';
import { initializeWebRTC } from './initialize-webrtc';
import { MOCK_PROMPT, MOCK_PROMPT_ID, SECOND_PROMPT, USB_CAMERA_SOURCE } from './mocks';

const waitForSAM = async (locator: Locator) => {
    await expect(locator.getByText('Processing image, please wait...')).toBeVisible({
        timeout: ANNOTATOR_PAGE_TIMEOUT,
    });
    await expect(locator.getByText('Processing image, please wait...')).toBeHidden({
        timeout: ANNOTATOR_PAGE_TIMEOUT,
    });
};

test.describe('Prompt', () => {
    test('Prompt flow', async ({ network, page, context, streamPage, annotatorPage, promptPage }) => {
        test.setTimeout(ANNOTATOR_PAGE_TIMEOUT);
        await initializeWebRTC({ page, context, network });

        const labels = registerApiLabels({ network });

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
        const promptPageFullScreen = new PromptPage(page, annotatorPageFullScreen.getScope());

        await test.step('Waits for SAM to load', async () => {
            await waitForSAM(annotatorPageFullScreen.getScope());
        });

        await test.step('Adds a label', async () => {
            const labelsPage = new LabelsPage(page, annotatorPageFullScreen.getScope());
            const labelName = 'Label 1';

            await labelsPage.showDialog();
            await labelsPage.addLabel(labelName);
            await expect(labelsPage.getLabel(labelName)).toBeVisible();
        });

        await test.step('Adds an annotation', async () => {
            await expect(promptPageFullScreen.savePromptButton).toBeDisabled();

            await annotatorPageFullScreen.addAnnotation();

            await expectToHaveAnnotations({ annotatorPage: annotatorPageFullScreen });
            await expect(promptPageFullScreen.savePromptButton).toBeEnabled();
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

            await promptPageFullScreen.savePrompt();

            await expect(promptPage.thumbnail).toHaveCount(1);
        });

        await test.step('Edits prompt', async () => {
            // Create a second prompt (we already have one from previous steps)
            await streamPage.captureFrame();
            const annotatorFullScreenEdit = annotatorPage.getFullScreen();

            await expect(annotatorFullScreenEdit.getCapturedFrame()).toBeVisible();

            await annotatorFullScreenEdit.addAnnotation();
            const promptPageFullScreenEdit = new PromptPage(page, annotatorFullScreenEdit.getScope());

            const mockPrompt = getMockedVisualPromptItem({
                ...MOCK_PROMPT,
                annotations: MOCK_PROMPT.annotations.map((annotation) => ({
                    ...annotation,
                    label_id: labels[0].id,
                })),
            });

            const secondMockPrompt = getMockedVisualPromptItem({
                ...SECOND_PROMPT,
                annotations: MOCK_PROMPT.annotations.map((annotation) => ({
                    ...annotation,
                    label_id: labels[0].id,
                })),
            });

            network.use(
                http.get('/api/v1/projects/{project_id}/prompts', ({ response }) => {
                    return response(200).json({
                        prompts: [mockPrompt, secondMockPrompt],
                        pagination: {
                            total: 2,
                            count: 2,
                            offset: 0,
                            limit: 10,
                        },
                    });
                })
            );

            await promptPageFullScreenEdit.savePrompt();
            await expect(promptPage.thumbnail).toHaveCount(2);

            // Edit the first prompt
            network.use(
                http.get('/api/v1/projects/{project_id}/prompts/{prompt_id}', ({ response }) => {
                    return response(200).json(mockPrompt);
                })
            );

            await promptPage.editPrompt(MOCK_PROMPT_ID);

            await waitForSAM(annotatorPage.getScope());

            // Add an annotation
            await annotatorPage.addAnnotation();

            network.use(
                http.put('/api/v1/projects/{project_id}/prompts/{prompt_id}', ({ response }) => {
                    return response(200).json(mockPrompt);
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

            let promptIdToBeDeleted = null;

            network.use(
                http.delete('/api/v1/projects/{project_id}/prompts/{prompt_id}', async ({ response, params }) => {
                    promptIdToBeDeleted = params.prompt_id;
                    return response(204).empty();
                })
            );

            await promptPage.deletePrompt(MOCK_PROMPT_ID);

            await expect(promptPage.thumbnail).toHaveCount(1);
            expect(promptIdToBeDeleted).toBe(MOCK_PROMPT_ID);
        });
    });

    test('Shows captured frame when there is already a prompt in canvas', async ({
        network,
        page,
        context,
        streamPage,
        promptPage,
        annotatorPage,
    }) => {
        await initializeWebRTC({ page, context, network });

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
            http.get('/api/v1/projects/{project_id}/labels', ({ response }) => {
                return response(200).json({
                    labels: [
                        {
                            id: 'label-1',
                            name: 'Label 1',
                            color: '#FF0000',
                        },
                    ],
                    pagination: {
                        total: 1,
                        count: 1,
                        offset: 0,
                        limit: 10,
                    },
                });
            }),
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

        await page.goto('/');

        await streamPage.startStream();

        await expect(promptPage.thumbnail).toHaveCount(1);

        await promptPage.editPrompt(MOCK_PROMPT_ID);

        await waitForSAM(annotatorPage.getScope());

        expect(page.url()).toContain(`promptId=${MOCK_PROMPT_ID}`);
        await expect(promptPage.getCapturedFrame(MOCK_PROMPT.frame_id)).toBeVisible();

        const FRAME_ID = '123';

        network.use(
            http.post('/api/v1/projects/{project_id}/frames', ({ response }) =>
                response(201).json({ frame_id: FRAME_ID })
            )
        );

        await streamPage.captureFrame();

        const promptPageFullScreen = new PromptPage(page, annotatorPage.getScope());

        await expect(promptPageFullScreen.getCapturedFrame(FRAME_ID)).toBeVisible();
        expect(page.url()).not.toContain(`promptId=`);
    });
});
