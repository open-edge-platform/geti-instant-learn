/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';

export const useAvailableDatasets = (notifyError = true) => {
    return $api.useQuery('get', '/api/v1/system/datasets', undefined, {
        meta: {
            error: {
                notify: notifyError,
            },
        },
        select: (data) => data.datasets,
    });
};
