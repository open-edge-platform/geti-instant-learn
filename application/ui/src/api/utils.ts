/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { fromOpenApi } from '@mswjs/source/open-api';
import { createOpenApiHttp } from 'openapi-msw';

import { type paths } from './openapi-spec';
import spec from './openapi-spec.json' with { type: 'json' };

const http = createOpenApiHttp<paths>({
    baseUrl: process.env.PUBLIC_API_URL ?? 'http://localhost:9100',
});

const handlers = await fromOpenApi(JSON.stringify(spec));

export { handlers, http };
