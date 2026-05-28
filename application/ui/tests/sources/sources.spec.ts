/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@/test-fixtures';

import { SourcesListType } from '../../src/api';
import { mockSourcesResponse, USB_SOURCE, VIDEO_SOURCE } from './mocks';

test.describe('Sources', () => {
    test('Creates a video file source', async ({ network, page, sourcesPage }) => {
        let sources: SourcesListType['sources'] = [];

        network.use(
            http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                return response(200).json(mockSourcesResponse(sources));
            }),
            http.post('/api/v1/projects/{project_id}/sources', ({ response }) => {
                sources = [VIDEO_SOURCE];
                return response(201).json(VIDEO_SOURCE);
            })
        );

        await sourcesPage.goto();
        await sourcesPage.openPipelineConfiguration();

        await test.step('Open Video file panel', async () => {
            await sourcesPage.openSourceTypePanel('Video file');
        });

        await test.step('Fill in path and submit', async () => {
            const panel = page.getByLabel('Video file');
            await panel.getByRole('textbox', { name: 'File path' }).fill('/home/user/video.mp4');
            await panel.getByRole('button', { name: 'Apply' }).click();
        });

        await test.step('Existing sources list appears', async () => {
            await expect(sourcesPage.addNewSourceButton).toBeVisible();
        });
    });

    test('Edits a video file source', async ({ network, page, sourcesPage }) => {
        const updatedSource = { ...VIDEO_SOURCE, config: { ...VIDEO_SOURCE.config, video_path: '/new/path.mp4' } };

        network.use(
            http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                return response(200).json(mockSourcesResponse([VIDEO_SOURCE]));
            }),
            http.put('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) => {
                return response(200).json(updatedSource);
            })
        );

        await sourcesPage.goto();
        await sourcesPage.openPipelineConfiguration();
        await sourcesPage.openSourceActions('Video file');
        await sourcesPage.selectAction('Edit');

        await expect(page.getByText('Edit input source')).toBeVisible();
        await page.getByRole('textbox', { name: 'File path' }).fill('/new/path.mp4');
        await page.getByRole('button', { name: 'Save', exact: true }).click();

        await expect(sourcesPage.addNewSourceButton).toBeVisible();
    });

    test('Connects an inactive source from the action menu', async ({ network, page, sourcesPage }) => {
        let sources: SourcesListType['sources'] = [USB_SOURCE];

        network.use(
            http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                return response(200).json(mockSourcesResponse(sources));
            }),
            http.put('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) => {
                sources = [{ ...USB_SOURCE, active: true }];
                return response(200).json({ ...USB_SOURCE, active: true });
            })
        );

        await sourcesPage.goto();
        await sourcesPage.openPipelineConfiguration();
        await sourcesPage.openSourceActions('USB Camera');
        await sourcesPage.selectAction('Connect');

        await sourcesPage.openSourceActions('USB Camera');
        await expect(page.getByRole('menuitem', { name: 'Connect' })).toBeHidden();
    });

    test('Deletes a source and returns to source type list when it was the last source', async ({
        network,
        page,
        sourcesPage,
    }) => {
        let sources: SourcesListType['sources'] = [VIDEO_SOURCE];

        network.use(
            http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                return response(200).json(mockSourcesResponse(sources));
            }),
            http.delete('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) => {
                sources = [];
                return response(204).empty();
            })
        );

        await sourcesPage.goto();
        await sourcesPage.openPipelineConfiguration();
        await sourcesPage.openSourceActions('Video file');
        await sourcesPage.selectAction('Delete');

        await expect(page.getByRole('button', { name: 'Video file' })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Image folder' })).toBeVisible();
    });
});
