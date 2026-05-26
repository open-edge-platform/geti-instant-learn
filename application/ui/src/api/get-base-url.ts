/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

export const getBaseUrl = async () => {
    return import.meta.env.PUBLIC_API_URL ?? '';
};
