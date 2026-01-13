/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType } from '@geti-prompt/api';
import { expect, http, test } from '@geti-prompt/test-fixtures';
import { NetworkFixture } from '@msw/playwright';
import { Page } from '@playwright/test';

import { initializeWebRTC } from '../prompt/initialize-webrtc';
import { StreamPage } from '../prompt/stream-page';
import { registerApiLabels, USB_CAMERA_SOURCE } from './mocks';

const setupLabels = async ({
    network,
    labels = [],
    page,
    streamPage,
}: {
    network: NetworkFixture;
    page: Page;
    streamPage: StreamPage;
    labels?: LabelType[];
}) => {
    registerApiLabels({ network, defaultLabels: labels });

    await page.goto('/');

    await streamPage.startStream();
    await streamPage.captureFrame();
};

test.describe('Labels', () => {
    test.beforeEach(async ({ network, context, page }) => {
        await initializeWebRTC({ network, context, page });

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
    });

    test('Creates label', async ({ page, streamPage, network, labelsPage }) => {
        await setupLabels({ network, page, streamPage });

        const newLabelName = 'Cool label';

        await labelsPage.showDialog();
        await labelsPage.addLabel(newLabelName);

        await expect(labelsPage.getLabel(newLabelName)).toBeVisible();
    });

    test('Deletes label', async ({ page, streamPage, network, labelsPage }) => {
        const label: LabelType = {
            id: '1',
            name: 'Cool label',
            color: '#000000',
        };
        await setupLabels({ network, page, streamPage, labels: [label] });

        await expect(labelsPage.getLabel(label.name)).toBeVisible();
        await labelsPage.deleteLabel(label.name);
        await expect(labelsPage.getLabel(label.name)).toBeHidden();
    });

    test('Updates label', async ({ page, streamPage, network, labelsPage }) => {
        const label: LabelType = {
            id: '1',
            name: 'Cool label',
            color: '#000000',
        };
        const newLabelName = 'New label name';

        await setupLabels({ network, page, streamPage, labels: [label] });

        await expect(labelsPage.getLabel(label.name)).toBeVisible();

        await labelsPage.updateLabelName(label.name, newLabelName);

        await expect(labelsPage.getLabel(label.name)).toBeHidden();
        await expect(labelsPage.getLabel(newLabelName)).toBeVisible();

        await labelsPage.enterEditLabelMode(newLabelName);
        await labelsPage.openColorPicker();

        const colorWithoutHash = label.color.replace('#', '');

        await expect(labelsPage.getColorInput()).toHaveValue(colorWithoutHash);

        await labelsPage.changeColor();

        await expect(labelsPage.getColorInput()).not.toHaveValue(colorWithoutHash);
    });
});
