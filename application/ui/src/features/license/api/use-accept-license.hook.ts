/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';

export const useAcceptLicense = () => {
    return $api.useMutation('post', '/api/v1/system/license/accept', {
        meta: {
            invalidates: [['get', '/health']],
            error: {
                notify: true,
            },
        },
    });
};
