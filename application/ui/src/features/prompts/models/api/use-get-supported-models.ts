/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';

export type { SupportedModelMetadataType as SupportedModelMetadata, SupportedPromptType } from '@/api';

export const useGetSupportedModels = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/system/supported-models', {
        params: { query: { offset: 0, limit: 20 } },
    });

    return data.models;
};
