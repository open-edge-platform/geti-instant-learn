/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@/test-fixtures';

import { SourcesListType } from '../../src/api';
import {
    DATASET_SOURCE,
    DATASETS_RESPONSE,
    EMPTY_DATASETS_RESPONSE,
    IMAGES_SOURCE,
    mockSourcesResponse,
    USB_CAMERAS_RESPONSE,
    USB_SOURCE,
    VIDEO_SOURCE,
} from './mocks';

test.describe('Sources', () => {
    test.describe('Initial state', () => {
        test('Shows source type options when no sources exist', async ({ page, sourcesPage }) => {
            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();

            await expect(page.getByRole('button', { name: 'USB Camera' })).toBeVisible();
            await expect(page.getByRole('button', { name: 'Image folder' })).toBeVisible();
            await expect(page.getByRole('button', { name: 'Video file' })).toBeVisible();
        });

        test('Does not show Sample dataset option when no datasets are available', async ({
            network,
            page,
            sourcesPage,
        }) => {
            network.use(
                http.get('/api/v1/system/datasets', ({ response }) => {
                    return response(200).json(EMPTY_DATASETS_RESPONSE);
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();

            await expect(page.getByRole('button', { name: 'Sample dataset' })).toBeHidden();
        });
    });

    test.describe('Create sources', () => {
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

        test('Creates an image folder source', async ({ network, page, sourcesPage }) => {
            let sources: SourcesListType['sources'] = [];

            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse(sources));
                }),
                http.post('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    sources = [IMAGES_SOURCE];
                    return response(201).json(IMAGES_SOURCE);
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceTypePanel('Image folder');

            const panel = page.getByLabel('Image folder');
            await panel.getByRole('textbox', { name: 'Folder path' }).fill('/home/user/images');
            await panel.getByRole('button', { name: 'Apply' }).click();

            await expect(sourcesPage.addNewSourceButton).toBeVisible();
        });

        test('Creates a USB camera source', async ({ network, page, sourcesPage }) => {
            let sources: SourcesListType['sources'] = [];

            network.use(
                http.get('/api/v1/system/source-types/{source_type}/sources', ({ response }) => {
                    return response(200).json(USB_CAMERAS_RESPONSE);
                }),
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse(sources));
                }),
                http.post('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    sources = [USB_SOURCE];
                    return response(201).json(USB_SOURCE);
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceTypePanel('USB Camera');

            await expect(page.getByRole('button', { name: 'Webcam HD' })).toBeVisible();
            await page.getByRole('button', { name: 'Apply' }).click();

            await expect(sourcesPage.addNewSourceButton).toBeVisible();
        });

        test('Shows "No USB Cameras found" when no USB devices are available', async ({
            network,
            page,
            sourcesPage,
        }) => {
            network.use(
                http.get('/api/v1/system/source-types/{source_type}/sources', ({ response }) => {
                    return response(200).json([]);
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceTypePanel('USB Camera');

            await expect(page.getByText('No USB Cameras found')).toBeVisible();
        });

        test('Creates a sample dataset source', async ({ network, page, sourcesPage }) => {
            let sources: SourcesListType['sources'] = [];

            network.use(
                http.get('/api/v1/system/datasets', ({ response }) => {
                    return response(200).json(DATASETS_RESPONSE);
                }),
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse(sources));
                }),
                http.post('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    sources = [DATASET_SOURCE];
                    return response(201).json(DATASET_SOURCE);
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceTypePanel('Sample dataset');

            await expect(page.getByRole('heading', { name: 'Sample Dataset 1', level: 3 })).toBeVisible();
            await page.getByRole('button', { name: 'Apply' }).click();

            await expect(sourcesPage.addNewSourceButton).toBeVisible();
        });

        test('Apply button is disabled when video file path is empty', async ({ page, sourcesPage }) => {
            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceTypePanel('Video file');

            const panel = page.getByLabel('Video file');
            await expect(panel.getByRole('button', { name: 'Apply' })).toBeDisabled();
        });

        test('Apply button is disabled when image folder path is empty', async ({ page, sourcesPage }) => {
            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceTypePanel('Image folder');

            const panel = page.getByLabel('Image folder');
            await expect(panel.getByRole('button', { name: 'Apply' })).toBeDisabled();
        });
    });

    test.describe('Existing sources list', () => {
        test('Shows existing sources', async ({ network, page, sourcesPage }) => {
            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([VIDEO_SOURCE]));
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();

            await expect(page.getByText('Video file')).toBeVisible();
            await expect(page.getByText('/home/user/video.mp4')).toBeVisible();
        });

        test('Active source action menu shows Edit and Delete but not Connect', async ({
            network,
            page,
            sourcesPage,
        }) => {
            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([{ ...VIDEO_SOURCE, active: true }]));
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceActions('Video file');

            await expect(page.getByRole('menuitem', { name: 'Edit' })).toBeVisible();
            await expect(page.getByRole('menuitem', { name: 'Delete' })).toBeVisible();
            await expect(page.getByRole('menuitem', { name: 'Connect' })).toBeHidden();
        });

        test('Inactive source action menu shows Connect, Edit, and Delete', async ({ network, page, sourcesPage }) => {
            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([{ ...VIDEO_SOURCE, active: false }]));
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceActions('Video file');

            await expect(page.getByRole('menuitem', { name: 'Connect' })).toBeVisible();
            await expect(page.getByRole('menuitem', { name: 'Edit' })).toBeVisible();
            await expect(page.getByRole('menuitem', { name: 'Delete' })).toBeVisible();
        });

        test('Hides source type from Add new source when that type already exists', async ({
            network,
            page,
            sourcesPage,
        }) => {
            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([VIDEO_SOURCE]));
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.addNewSourceButton.click();

            await expect(page.getByRole('button', { name: 'Video file' })).toBeHidden();
            await expect(page.getByRole('button', { name: 'Image folder' })).toBeVisible();
        });

        test('Hides Add new source button when all source types are present', async ({ network, sourcesPage }) => {
            network.use(
                http.get('/api/v1/system/datasets', ({ response }) => {
                    return response(200).json(DATASETS_RESPONSE);
                }),
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(
                        mockSourcesResponse([VIDEO_SOURCE, IMAGES_SOURCE, USB_SOURCE, DATASET_SOURCE])
                    );
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();

            await expect(sourcesPage.addNewSourceButton).toBeHidden();
        });
    });

    test.describe('Edit sources', () => {
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

        test('Edits an image folder source', async ({ network, page, sourcesPage }) => {
            const updatedSource = {
                ...IMAGES_SOURCE,
                config: { ...IMAGES_SOURCE.config, images_folder_path: '/new/images' },
            };

            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([IMAGES_SOURCE]));
                }),
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) => {
                    return response(200).json(updatedSource);
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceActions('Images folder');
            await sourcesPage.selectAction('Edit');

            await expect(page.getByText('Edit input source')).toBeVisible();
            await page.getByRole('textbox', { name: 'Folder path' }).fill('/new/images');
            await page.getByRole('button', { name: 'Save', exact: true }).click();

            await expect(sourcesPage.addNewSourceButton).toBeVisible();
        });

        test('Save button is disabled when path is unchanged', async ({ network, page, sourcesPage }) => {
            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([VIDEO_SOURCE]));
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceActions('Video file');
            await sourcesPage.selectAction('Edit');

            await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();
        });

        test('Shows Save and Save & Connect buttons for inactive source', async ({ network, page, sourcesPage }) => {
            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([{ ...VIDEO_SOURCE, active: false }]));
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceActions('Video file');
            await sourcesPage.selectAction('Edit');

            await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeVisible();
            await expect(page.getByRole('button', { name: 'Save & Connect' })).toBeVisible();
        });

        test('Does not show Save & Connect for active source', async ({ network, page, sourcesPage }) => {
            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse([{ ...VIDEO_SOURCE, active: true }]));
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceActions('Video file');
            await sourcesPage.selectAction('Edit');

            await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeVisible();
            await expect(page.getByRole('button', { name: 'Save & Connect' })).toBeHidden();
        });
    });

    test.describe('Connect source', () => {
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
    });

    test.describe('Delete sources', () => {
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

        test('Deletes one of multiple sources and keeps existing list visible', async ({ network, sourcesPage }) => {
            let sources: SourcesListType['sources'] = [VIDEO_SOURCE, IMAGES_SOURCE];

            network.use(
                http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                    return response(200).json(mockSourcesResponse(sources));
                }),
                http.delete('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) => {
                    sources = [IMAGES_SOURCE];
                    return response(204).empty();
                })
            );

            await sourcesPage.goto();
            await sourcesPage.openPipelineConfiguration();
            await sourcesPage.openSourceActions('Video file');
            await sourcesPage.selectAction('Delete');

            await expect(sourcesPage.addNewSourceButton).toBeVisible();
            await expect(sourcesPage.getSourceCard('Images folder')).toBeVisible();
        });
    });
});
