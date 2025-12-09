/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import createFetchClient from 'openapi-fetch';
import createClient from 'openapi-react-query';

import type { paths } from './openapi-spec';

const getBaseURL = () => {
    if (import.meta.env?.VITEST) {
        // or if playwright CI
        return 'http://localhost:9100';
    }

    return '/';
};

export const client = createFetchClient<paths>({
    baseUrl: getBaseURL(),
    fetch: (options) => fetch(options),
});

export const $api = createClient(client);
