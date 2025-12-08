/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createNetworkFixture, NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';
import { HttpResponse } from 'msw';

import { handlers, http } from '../src/api/utils';
import { AnnotatorPage } from './annotator/annotator-page';
import { PromptPage } from './annotator/prompt-page';
import { LabelsPage } from './labels/labels-page';
import { ProjectPage } from './projects/projects-page';
import { StreamPage } from './prompt/stream-page';

interface Fixtures {
    network: NetworkFixture;
    streamPage: StreamPage;
    labelsPage: LabelsPage;
    annotatorPage: AnnotatorPage;
    projectPage: ProjectPage;
    promptPage: PromptPage;
}

const test = testBase.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: [
            ...handlers,
            http.get('/api/v1/projects', ({ response }) => {
                return response(200).json({
                    projects: [
                        {
                            id: '1',
                            name: 'Project #1',
                            active: true,
                        },
                    ],
                    pagination: { total: 1, count: 1, offset: 0, limit: 10 },
                });
            }),
            http.get('/api/v1/projects/{project_id}', ({ response }) => {
                return response(200).json({
                    id: '1',
                    name: 'Project #1',
                    active: true,
                });
            }),
            http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                return response(200).json({
                    sources: [],
                    pagination: {
                        count: 0,
                        total: 0,
                        limit: 10,
                        offset: 0,
                    },
                });
            }),
            http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                return response(200).json({
                    sinks: [],
                    pagination: {
                        count: 0,
                        total: 0,
                        limit: 10,
                        offset: 0,
                    },
                });
            }),
            http.get('/api/v1/projects/{project_id}/labels', ({ response }) => {
                return response(200).json({
                    labels: [],
                    pagination: {
                        total: 0,
                        count: 0,
                        offset: 0,
                        limit: 10,
                    },
                });
            }),
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
            }),
            http.post('/api/v1/projects/{project_id}/prompts', ({ response }) => {
                return response(201).json({
                    id: 'prompt-id',
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
                                    {
                                        x: 0.1,
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
                });
            }),
            http.delete('/api/v1/projects/{project_id}/prompts/{prompt_id}', () => {
                return HttpResponse.json({}, { status: 204 });
            }),
            http.put('/api/v1/projects/{project_id}/prompts/{prompt_id}', async ({ request, params }) => {
                const body = (await request.json()) as Record<string, unknown>;
                const promptId = params.prompt_id as string;

                return HttpResponse.json({ id: promptId, ...body } as never);
            }),
            http.get('/api/v1/projects/{project_id}/models', ({ response }) => {
                return response(200).json({
                    models: [],
                    pagination: {
                        total: 0,
                        count: 0,
                        offset: 0,
                        limit: 10,
                    },
                });
            }),
        ],
    }),
    streamPage: async ({ page }, use) => {
        const streamPage = new StreamPage(page);
        await use(streamPage);
    },
    labelsPage: async ({ page }, use) => {
        const labelsPage = new LabelsPage(page);
        await use(labelsPage);
    },
    annotatorPage: async ({ page }, use) => {
        const annotatorPage = new AnnotatorPage(page);
        await use(annotatorPage);
    },
    projectPage: async ({ page }, use) => {
        const projectPage = new ProjectPage(page);
        await use(projectPage);
    },
    promptPage: async ({ page }, use) => {
        const promptPage = new PromptPage(page);
        await use(promptPage);
    },
});

export { expect, test, http };
