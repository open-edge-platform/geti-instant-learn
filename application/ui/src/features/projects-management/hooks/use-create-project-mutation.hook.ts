/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';

export const useCreateProjectMutation = () => {
    return $api.useMutation('post', '/api/v1/projects', {
        meta: {
            invalidates: [['get', '/api/v1/projects']],
        },
    });
};
