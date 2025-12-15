/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@geti-prompt/test-fixtures';

test.describe('Health Check', () => {
    test('Shows loading when heath check is pending', async ({ page, network }) => {
        let i = 0;
        network.use(
            http.get('/health', ({ response }) => {
                i += 1;
                if (i < 3) {
                    // @ts-expect-error We want to mock behavior when server is not ready yet
                    return response(500).json({});
                }

                return response(200).json({ status: 'ok' });
            })
        );

        await page.goto('/');

        await expect(page.getByRole('progressbar')).toBeVisible();

        await expect(page.getByRole('progressbar')).toBeHidden();
    });
});
