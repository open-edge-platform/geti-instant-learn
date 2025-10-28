/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createNetworkFixture, NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';

import { handlers, http } from '../src/api/utils';

interface Fixtures {
    network: NetworkFixture;
}

const test = testBase.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: [
            ...handlers,
            http.get('/api/v1/projects', ({ response }) => {
                return response(200).json({
                    projects: [
                        {
                            id: '1',
                            name: 'Project #1',
                            active: true,
                        },
                    ],
                    pagination: { total: 1, count: 1, offset: 0, limit: 10 },
                });
            }),
            http.get('/api/v1/projects/{project_id}', ({ response }) => {
                return response(200).json({
                    id: '1',
                    name: 'Project #1',
                    active: true,
                });
            }),
            http.get('/api/v1/projects/{project_id}/sources', ({ response }) => {
                return response(200).json({
                    sources: [],
                });
            }),
            http.get('/api/v1/projects/{project_id}/labels', ({ response }) => {
                return response(200).json({
                    labels: [],
                    pagination: {
                        total: 0,
                        count: 0,
                        offset: 0,
                        limit: 10,
                    },
                });
            }),
        ],
    }),
});

export { expect, test, http };
