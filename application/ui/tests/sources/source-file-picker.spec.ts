/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, test } from '@/test-fixtures';

test.describe('Source file picker fields', () => {
    test.beforeEach(async ({ page }) => {
        await page.addInitScript(() => {
            /* eslint-disable no-underscore-dangle */
            const runtime = window as typeof window & {
                __TAURI__?: {
                    core: {
                        invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
                    };
                };
                __TAURI_INTERNALS__?: {
                    invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
                    transformCallback: () => number;
                    unregisterCallback: () => void;
                };
            };

            const invoke = async <T>(cmd: string, args?: Record<string, unknown>): Promise<T> => {
                if (cmd === 'get_public_api_url') {
                    return '' as T;
                }

                if (cmd === 'plugin:dialog|open') {
                    const options = args?.options as { directory?: boolean } | undefined;

                    return (options?.directory === true ? '/home/user/images' : '/home/user/video.mp4') as T;
                }

                return null as T;
            };

            runtime.__TAURI__ = { core: { invoke } };
            runtime.__TAURI_INTERNALS__ = {
                invoke,
                transformCallback: () => 1,
                unregisterCallback: () => undefined,
            };
            /* eslint-enable no-underscore-dangle */
        });
    });

    test('updates the video file path after selecting a file from the dialog', async ({ page, network: _ }) => {
        await page.goto('/');

        await page.getByRole('button', { name: 'Pipeline configuration' }).click();

        await page.getByRole('button', { name: 'Video file' }).click();

        const videoFilePanel = page.getByLabel('Video file');

        await videoFilePanel.getByRole('button', { name: 'Browse' }).click();

        await expect(videoFilePanel.getByRole('textbox', { name: 'File path' })).toHaveValue('/home/user/video.mp4');
    });

    test('updates the image folder path after selecting a folder from the dialog', async ({ page, network: _ }) => {
        await page.goto('/');

        await page.getByRole('button', { name: 'Pipeline configuration' }).click();

        await page.getByRole('button', { name: 'Image folder' }).click();

        const imageFolderPanel = page.getByLabel('Image folder');

        await imageFolderPanel.getByRole('button', { name: 'Browse' }).click();

        await expect(imageFolderPanel.getByRole('textbox', { name: 'Folder path' })).toHaveValue('/home/user/images');
    });
});
