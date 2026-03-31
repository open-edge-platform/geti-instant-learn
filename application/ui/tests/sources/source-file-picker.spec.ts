/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, test } from '@/test-fixtures';
import type { Page } from '@playwright/test';

const openSourceTypePanel = async (page: Page, sourceType: 'Video file' | 'Image folder') => {
    const pipelineConfigurationButton = page.getByRole('button', { name: 'Pipeline configuration' });

    await expect(pipelineConfigurationButton).toBeVisible();
    await pipelineConfigurationButton.click();

    const addNewSourceButton = page.getByRole('button', { name: 'Add new source' });

    if ((await addNewSourceButton.count()) > 0) {
        await addNewSourceButton.click();
    }

    const sourceTypeButton = page.getByRole('button', { name: sourceType });
    await expect(sourceTypeButton).toBeVisible();
    await sourceTypeButton.click();
};

test.describe('Source file picker fields', () => {
    test.beforeEach(async ({ page }) => {
        await page.addInitScript((apiUrl: string) => {
            const invoke = async (cmd: string, args?: Record<string, unknown>) => {
                if (cmd === 'get_public_api_url') return apiUrl;
                if (cmd === 'plugin:dialog|open') {
                    return (args?.options as { directory?: boolean })?.directory === true
                        ? '/home/user/images'
                        : '/home/user/video.mp4';
                }

                return null;
            };

            Object.assign(window, {
                __TAURI__: { core: { invoke } },
                __TAURI_INTERNALS__: { invoke, transformCallback: () => 1, unregisterCallback: () => undefined },
            });
        }, process.env.PUBLIC_API_URL ?? '');
    });

    test('updates the video file path after selecting a file from the dialog', async ({ page }) => {
        await page.goto('/');

        await openSourceTypePanel(page, 'Video file');

        const videoFilePanel = page.getByLabel('Video file');

        await videoFilePanel.getByRole('button', { name: 'Browse' }).click();

        await expect(videoFilePanel.getByRole('textbox', { name: 'File path' })).toHaveValue('/home/user/video.mp4');
    });

    test('updates the image folder path after selecting a folder from the dialog', async ({ page }) => {
        await page.goto('/');

        await openSourceTypePanel(page, 'Image folder');

        const imageFolderPanel = page.getByLabel('Image folder');

        await imageFolderPanel.getByRole('button', { name: 'Browse' }).click();

        await expect(imageFolderPanel.getByRole('textbox', { name: 'Folder path' })).toHaveValue('/home/user/images');
    });
});
