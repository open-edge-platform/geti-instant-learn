/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ProjectsListType, ProjectType } from '@geti-prompt/api';
import { fromOpenApi } from '@mswjs/source/open-api';
import { HttpResponse } from 'msw';
import { createOpenApiHttp, OpenApiHttpHandlers } from 'openapi-msw';

import { type paths } from './openapi-spec';
import spec from './openapi-spec.json' with { type: 'json' };

const getOpenApiHttp = (): OpenApiHttpHandlers<paths> => {
    const http = createOpenApiHttp<paths>({
        baseUrl: process.env.PUBLIC_API_URL ?? 'http://localhost:9100',
    });

    return {
        ...http,
        post: (path, ...other) => {
            // @ts-expect-error MSW internal parsing function does not accept paths like
            // `/api/models/{model_name}:activate`
            // to get around this we escape the colon character with `\\`
            // @see https://github.com/mswjs/msw/discussions/739
            return http.post(path.replace('}:', '}\\:'), ...other);
        },
    };
};

const http = getOpenApiHttp();

const openApiHandlers = await fromOpenApi(JSON.stringify(spec).replace(/}:/g, '}//:'));

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

const handlers = [...openApiHandlers, ...initialHandlers];

export { handlers, http };
