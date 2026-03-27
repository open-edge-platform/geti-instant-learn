/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, test } from '@/test-fixtures';

test.describe('Source file picker fields', () => {
    test('shows browse controls for video and image-folder sources in web runtime', async ({ page }) => {
        await page.goto('/');

        await page.getByRole('button', { name: 'Pipeline configuration' }).click();

        await page.getByRole('button', { name: 'Video file' }).click();

        const videoFilePanel = page.getByLabel('Video file');

        await expect(videoFilePanel.getByRole('textbox', { name: 'File path' })).toBeVisible();
        await expect(videoFilePanel.getByRole('button', { name: 'Browse' })).toBeDisabled();

        await page.getByRole('button', { name: 'Image folder' }).click();

        const imageFolderPanel = page.getByLabel('Image folder');

        await expect(imageFolderPanel.getByRole('textbox', { name: 'Folder path' })).toBeVisible();
        await expect(imageFolderPanel.getByRole('button', { name: 'Browse' })).toBeDisabled();
    });
});
