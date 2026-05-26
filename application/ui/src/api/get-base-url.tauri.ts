/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable no-underscore-dangle */
export const getBaseUrl = async () => {
    return window.__TAURI__!.core!.invoke<string>('get_public_api_url');
};
/* eslint-enable no-underscore-dangle */
