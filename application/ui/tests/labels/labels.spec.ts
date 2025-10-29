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
import { LabelsPage } from './labels-page';
import { registerApiLabels, WEBCAM_SOURCE } from './mocks';

const setupLabels = async ({
    network,
    labels = [],
    page,
}: {
    network: NetworkFixture;
    page: Page;
    labels?: LabelType[];
}) => {
    const labelsPage = new LabelsPage(page);
    const streamPage = new StreamPage(page);

    registerApiLabels({ network, defaultLabels: labels });

    await page.goto('/');

    await streamPage.startStream();
    await streamPage.captureFrame();

    return {
        labelsPage,
    };
};

test.describe('Labels', () => {
    test.beforeEach(async ({ network, context, page }) => {
        await initializeWebRTC({ network, context, page });

        network.use(
            http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                return response(200).json({ sources: [WEBCAM_SOURCE] });
            }),

            http.put('/api/v1/projects/{project_id}/sources/{source_id}', ({ response }) =>
                response(200).json(WEBCAM_SOURCE)
            )
        );
    });

    test('Creates label', async ({ page, network }) => {
        const { labelsPage } = await setupLabels({ network, page });

        const newLabelName = 'Cool label';

        await labelsPage.addLabel(newLabelName);

        await expect(labelsPage.getLabel(newLabelName)).toBeVisible();
    });

    test('Deletes label', async ({ page, network }) => {
        const label: LabelType = {
            id: '1',
            name: 'Cool label',
            color: '#000000',
        };

        const { labelsPage } = await setupLabels({ network, page, labels: [label] });

        await expect(labelsPage.getLabel(label.name)).toBeVisible();
        await labelsPage.deleteLabel(label.name);
        await expect(labelsPage.getLabel(label.name)).toBeHidden();
    });

    test('Updates label', async ({ page, network }) => {
        const label: LabelType = {
            id: '1',
            name: 'Cool label',
            color: '#000000',
        };
        const newLabelName = 'New label name';

        const { labelsPage } = await setupLabels({ network, page, labels: [label] });

        await expect(labelsPage.getLabel(label.name)).toBeVisible();

        await labelsPage.updateLabelName(label.name, newLabelName);

        await expect(labelsPage.getLabel(label.name)).toBeHidden();
        await expect(labelsPage.getLabel(newLabelName)).toBeVisible();
    });
});
