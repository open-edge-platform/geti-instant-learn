/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import createFetchClient from 'openapi-fetch';
import createClient from 'openapi-react-query';

import type { paths } from './openapi-spec';

export const baseUrl = import.meta.env.PUBLIC_API_BASE_URL || '';

export const client = createFetchClient<paths>({
    baseUrl,
    fetch: (options) => fetch(options),
});

export const $api = createClient(client);
