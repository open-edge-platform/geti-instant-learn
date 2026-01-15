/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import createFetchClient from 'openapi-fetch';
import createClient from 'openapi-react-query';

import type { paths } from './openapi-spec';

/* eslint no-underscore-dangle: ["error", { "allow": ["__TAURI__"] }]*/
/* eslint-disable  @typescript-eslint/no-explicit-any */
const invoke = (<any> window).__TAURI__?.core?.invoke;
let tauriPublicApiUrl = null;
if (invoke) {
    tauriPublicApiUrl = await invoke('get_public_api_url')
    console.info('Backend public API URL:', tauriPublicApiUrl);
}

export const client = createFetchClient<paths>({
    baseUrl: tauriPublicApiUrl || import.meta.env.PUBLIC_API_URL || '',
    fetch: (options) => fetch(options),
});

export const $api = createClient(client);
