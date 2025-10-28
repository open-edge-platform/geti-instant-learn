/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import '@testing-library/jest-dom/vitest';

import { ProjectsListType, ProjectType } from '@geti-prompt/api';
import { HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import fetchPolyfill, { Request as RequestPolyfill } from 'node-fetch';

import { handlers, http } from './api/utils';
import { queryClient } from './query-client/query-client';

const MOCKED_PROJECT_RESPONSE: ProjectType = {
    id: '1',
    name: 'Project #1',
    active: true,
};
const MOCKED_PROJECTS_LIST_RESPONSE: ProjectsListType = {
    projects: [MOCKED_PROJECT_RESPONSE],
    pagination: { total: 1, count: 1, offset: 0, limit: 10 },
};

const initialHandlers = [
    http.get('/api/v1/projects', () => {
        return HttpResponse.json(MOCKED_PROJECTS_LIST_RESPONSE);
    }),

    http.get('/api/v1/projects/{project_id}', () => {
        return HttpResponse.json(MOCKED_PROJECT_RESPONSE);
    }),
];

const server = setupServer(...handlers, ...initialHandlers);
export { http, server };

beforeAll(() => {
    server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
    server.resetHandlers();
    queryClient.clear();
});

afterAll(() => {
    server.close();
});

// Why we need these polyfills:
// https://github.com/reduxjs/redux-toolkit/issues/4966#issuecomment-3115230061
Object.defineProperty(global, 'fetch', {
    // MSW will overwrite this to intercept requests
    writable: true,
    value: fetchPolyfill,
});

Object.defineProperty(global, 'Request', {
    writable: false,
    value: RequestPolyfill,
});
