/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import type { components } from '@/api/openapi-spec';

export type SupportedModelMetadata = components['schemas']['SupportedModelMetadataSchema'];
export type SupportedPromptType = components['schemas']['SupportedPromptType'];

export const useGetSupportedModels = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/system/supported-models', {
        params: { query: { offset: 0, limit: 20 } },
    });

    return data.models;
};
