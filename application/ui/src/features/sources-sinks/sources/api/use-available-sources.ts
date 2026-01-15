/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, SourceType } from '@geti-prompt/api';

export const useAvailableSources = (sourceType: SourceType) => {
    return $api.useSuspenseQuery('get', '/api/v1/source-types/{source_type}/sources', {
        params: {
            path: {
                source_type: sourceType,
            },
        },
    });
};

export const usePrefetchAvailableSources = (sourceType: SourceType) => {
    return $api.useQuery('get', '/api/v1/source-types/{source_type}/sources', {
        params: {
            path: {
                source_type: sourceType,
            },
        },
    });
};
