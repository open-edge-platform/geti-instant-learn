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
                            id: '12',
                            name: 'Project #12',
                        },
                    ],
                });
            }),
            http.get('/api/v1/projects/{project_id}', ({ response }) => {
                return response(200).json({
                    id: '1',
                    name: 'Project #1',
                });
            }),
        ],
    }),
});

export { expect, test, http };
