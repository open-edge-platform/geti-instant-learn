/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@/test-fixtures';

import { paths } from '../../src/constants/paths';

test.describe('License agreement', () => {
    test('shows the license screen and accepts the license', async ({ page, network }) => {
        let licenseAccepted = false;

        network.use(
            http.get('/health', ({ response }) => {
                return response(200).json({
                    status: 'ok',
                    license_accepted: licenseAccepted,
                });
            }),
            http.post('/api/v1/license/accept', ({ response }) => {
                licenseAccepted = true;

                return response(200).json({ accepted: true });
            })
        );

        await test.step('license screen is shown when license is not accepted', async () => {
            await page.goto(paths.root({}));

            await expect(page.getByRole('heading', { name: /License Agreement/i })).toBeVisible();
            await expect(page.getByRole('link', { name: /SAM3 License/i })).toBeVisible();
            await expect(page.getByRole('link', { name: /DINOv3 License/i })).toBeVisible();
        });

        await test.step('Cancel button is disabled', async () => {
            await expect(page.getByRole('button', { name: /Cancel/i })).toBeDisabled();
        });

        await test.step('accepting the license loads the app', async () => {
            await page.getByRole('button', { name: /Accept and continue/i }).click();

            await expect(page.getByRole('heading', { name: /License Agreement/i })).toBeHidden();
        });
    });

    test('Cancel button shows a tooltip explaining it is disabled', async ({ page, network }) => {
        network.use(
            http.get('/health', ({ response }) => {
                return response(200).json({
                    status: 'ok',
                    license_accepted: false,
                });
            })
        );

        await page.goto(paths.root({}));

        await expect(page.getByRole('heading', { name: /License Agreement/i })).toBeVisible();

        await page.getByRole('button', { name: /Cancel/i }).hover();

        await expect(page.getByRole('tooltip')).toHaveText('You must accept the license to use this app');
    });
});
