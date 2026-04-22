/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import type { SchemaSupportedModelMetadataSchema, SchemaSupportedPromptType } from '@/api/openapi-spec';

export type SupportedModelMetadata = SchemaSupportedModelMetadataSchema;
export type SupportedPromptType = SchemaSupportedPromptType;

export const useGetSupportedModels = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/system/supported-models', {
        params: { query: { offset: 0, limit: 20 } },
    });

    return data.models;
};
