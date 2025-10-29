/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType, WebcamConfig } from '@geti-prompt/api';
import { expect, http, test } from '@geti-prompt/test-fixtures';
import { NetworkFixture } from '@msw/playwright';
import { HttpResponse } from 'msw';

import { initializeWebRTC } from '../prompt/initialize-webrtc';
import { StreamPage } from '../prompt/stream-page';
import { LabelsPage } from './labels-page';

const DEVICE_ID = 10;
const WEBCAM_SOURCE: WebcamConfig = {
    connected: true,
    id: 'webcam-id',
    config: {
        device_id: DEVICE_ID,
        source_type: 'webcam',
    },
};

const registerApiLabels = ({
    network,
    defaultLabels = [],
}: {
    network: NetworkFixture;
    defaultLabels?: LabelType[];
}) => {
    let labels = [...defaultLabels];

    network.use(
        http.get('/api/v1/projects/{project_id}/labels', () => {
            return HttpResponse.json({
                labels,
                pagination: { count: labels.length, total: labels.length, offset: 0, limit: 10 },
            });
        }),

        http.delete('/api/v1/projects/{project_id}/labels/{label_id}', ({ params }) => {
            const labelId = params.label_id;

            labels = labels.filter((label) => label.id !== labelId);

            return HttpResponse.json({}, { status: 204 });
        }),

        http.post('/api/v1/projects/{project_id}/labels', async ({ request }) => {
            const body = await request.json();

            labels.push(body as LabelType);

            return HttpResponse.json(body, { status: 201 });
        }),

        http.put('/api/v1/projects/{project_id}/labels/{label_id}', async ({ request, params }) => {
            const labelId = params.label_id;
            const body = await request.json();

            labels = labels.map((label) => (label.id === labelId ? ({ ...label, ...body } as LabelType) : label));

            return HttpResponse.json(body as LabelType, { status: 200 });
        })
    );
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
        const labelsPage = new LabelsPage(page);
        const streamPage = new StreamPage(page);

        registerApiLabels({ network });

        await page.goto('/');

        await streamPage.startStream();
        await streamPage.captureFrame();

        const newLabelName = 'Cool label';

        await labelsPage.addLabel(newLabelName);

        await expect(labelsPage.getLabel(newLabelName)).toBeVisible();
    });

    test('Deletes label', async ({ page, network }) => {
        const labelsPage = new LabelsPage(page);
        const streamPage = new StreamPage(page);

        const label: LabelType = {
            id: '1',
            name: 'Cool label',
            color: '#000000',
        };

        registerApiLabels({ network, defaultLabels: [label] });

        await page.goto('/');
        await streamPage.startStream();
        await streamPage.captureFrame();

        await expect(labelsPage.getLabel(label.name)).toBeVisible();
        await labelsPage.deleteLabel(label.name);
        await expect(labelsPage.getLabel(label.name)).toBeHidden();
    });

    test('Updates label', async ({ page, network }) => {
        const labelsPage = new LabelsPage(page);
        const streamPage = new StreamPage(page);

        const label: LabelType = {
            id: '1',
            name: 'Cool label',
            color: '#000000',
        };
        const newLabelName = 'New label name';

        registerApiLabels({ network, defaultLabels: [label] });

        await page.goto('/');
        await streamPage.startStream();
        await streamPage.captureFrame();

        await expect(labelsPage.getLabel(label.name)).toBeVisible();

        await labelsPage.updateLabelName(label.name, newLabelName);

        await expect(labelsPage.getLabel(label.name)).toBeHidden();
        await expect(labelsPage.getLabel(newLabelName)).toBeVisible();
    });
});
