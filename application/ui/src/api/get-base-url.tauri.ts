/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable no-underscore-dangle */
export const getBaseUrl = async () => {
    try {
        return (
            (await window.__TAURI__?.core?.invoke<string>('get_public_api_url')) ?? import.meta.env.PUBLIC_API_URL ?? ''
        );
    } catch {
        return import.meta.env.PUBLIC_API_URL ?? '';
    }
};
/* eslint-enable no-underscore-dangle */
