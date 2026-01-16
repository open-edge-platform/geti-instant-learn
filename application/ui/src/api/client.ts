/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import createFetchClient from 'openapi-fetch';
import createClient from 'openapi-react-query';

import type { paths } from './openapi-spec';

/* eslint-disable no-underscore-dangle */
const isTauriContext = (): boolean => typeof window.__TAURI__?.core?.invoke === 'function';

const getBaseUrl = async (): Promise<string> => {
    if (isTauriContext()) {
        const tauriApiUrl = await window.__TAURI__!.core!.invoke<string>('get_public_api_url');

        console.info('Backend public API URL:', tauriApiUrl);

        return tauriApiUrl;
    }

    return import.meta.env.PUBLIC_API_URL || '';
};
/* eslint-enable no-underscore-dangle */

export const baseUrl = await getBaseUrl();

export const client = createFetchClient<paths>({
    baseUrl,
    fetch: (options) => fetch(options),
});

export const $api = createClient(client);
