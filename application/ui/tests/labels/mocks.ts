/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType, WebcamSourceType } from '@geti-prompt/api';
import { http } from '@geti-prompt/test-fixtures';
import { NetworkFixture } from '@msw/playwright';
import { HttpResponse } from 'msw';

const DEVICE_ID = 10;
export const WEBCAM_SOURCE: WebcamSourceType = {
    active: true,
    id: 'webcam-id',
    config: {
        seekable: false,
        device_id: DEVICE_ID,
        source_type: 'webcam',
    },
};

export const registerApiLabels = ({
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
