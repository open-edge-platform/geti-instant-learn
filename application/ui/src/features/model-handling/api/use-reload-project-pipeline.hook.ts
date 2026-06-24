/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';

export const useReloadProjectPipeline = () => {
    return $api.useMutation('post', '/api/v1/projects/{project_id}/reload');
};
