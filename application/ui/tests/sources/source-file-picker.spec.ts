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

        await expect(page.getByRole('textbox', { name: 'File path' })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Browse' })).toBeDisabled();

        await page.getByRole('button', { name: 'Image folder' }).click();

        await expect(page.getByRole('textbox', { name: 'Folder path' })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Browse' })).toBeDisabled();
    });
});