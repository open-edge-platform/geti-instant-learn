/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import '@testing-library/jest-dom/vitest';

import {
    LabelListType,
    ModelListType,
    ProjectsListType,
    ProjectType,
    SinksListType,
    SourcesListType,
    VisualPromptListType,
} from '@geti-prompt/api';
import { queryClient } from '@geti-prompt/query-client';
import { HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import fetchPolyfill, { Request as RequestPolyfill } from 'node-fetch';

import { handlers, http } from './api/utils';

const MOCKED_PROJECT_RESPONSE: ProjectType = {
    id: '1',
    name: 'Project #1',
    active: true,
};
const MOCKED_PROJECTS_LIST_RESPONSE: ProjectsListType = {
    projects: [MOCKED_PROJECT_RESPONSE],
    pagination: { total: 1, count: 1, offset: 0, limit: 10 },
};
const MOCKED_LABELS_RESPONSE: LabelListType = {
    labels: [
        {
            id: 'test-id',
            name: 'test-label',
            color: 'red',
        },
    ],
    pagination: {
        count: 1,
        total: 1,
        offset: 0,
        limit: 10,
    },
};
const MOCKED_PROMPTS_RESPONSE: VisualPromptListType = {
    prompts: [],
    pagination: {
        count: 0,
        total: 0,
        offset: 0,
        limit: 10,
    },
};
const MOCKED_SOURCES_RESPONSE: SourcesListType = {
    sources: [],
    pagination: {
        count: 0,
        total: 0,
        limit: 10,
        offset: 0,
    },
};
const MOCKED_SINKS_RESPONSE: SinksListType = {
    sinks: [],
    pagination: {
        count: 0,
        total: 0,
        limit: 10,
        offset: 0,
    },
};
const MOCKED_MODELS_RESPONSE: ModelListType = {
    models: [
        {
            id: 'some-id',
            config: {
                confidence_threshold: 0.38,
                model_type: 'matcher',
                num_background_points: 2,
                num_foreground_points: 40,
                precision: 'bf16',
                sam_model: 'SAM-HQ-tiny',
                encoder_model: 'dinov3_large',
                use_mask_refinement: false,
            },
            active: true,
            name: 'Mega model',
        },
    ],
    pagination: {
        count: 0,
        total: 0,
        offset: 0,
        limit: 10,
    },
};

const initialHandlers = [
    http.get('/api/v1/projects', () => {
        return HttpResponse.json(MOCKED_PROJECTS_LIST_RESPONSE);
    }),

    http.get('/api/v1/projects/{project_id}', () => {
        return HttpResponse.json(MOCKED_PROJECT_RESPONSE);
    }),

    http.get('/api/v1/projects/{project_id}/labels', () => {
        return HttpResponse.json(MOCKED_LABELS_RESPONSE);
    }),

    http.get('/api/v1/projects/{project_id}/prompts', () => {
        return HttpResponse.json(MOCKED_PROMPTS_RESPONSE);
    }),

    http.get('/api/v1/projects/{project_id}/sources', () => {
        return HttpResponse.json(MOCKED_SOURCES_RESPONSE);
    }),

    http.get('/api/v1/projects/{project_id}/sinks', () => {
        return HttpResponse.json(MOCKED_SINKS_RESPONSE);
    }),

    http.get('/api/v1/projects/{project_id}/models', () => {
        return HttpResponse.json(MOCKED_MODELS_RESPONSE);
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

class CustomImageData {
    width: number;
    height: number;

    constructor(width: number, height: number) {
        this.width = width;
        this.height = height;
    }
}

global.ImageData = CustomImageData as typeof ImageData;

const IntersectionObserverMock = vi.fn(() => ({
    disconnect: vi.fn(),
    observe: vi.fn(),
    takeRecords: vi.fn(),
    unobserve: vi.fn(),
}));

vi.stubGlobal('IntersectionObserver', IntersectionObserverMock);
