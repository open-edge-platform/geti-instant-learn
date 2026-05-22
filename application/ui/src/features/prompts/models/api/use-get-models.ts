/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier, usePromptMode } from '@/hooks';

export const useGetModels = () => {
    const { projectId } = useProjectIdentifier();
    const [promptMode] = usePromptMode();

    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/models', {
        params: {
            path: { project_id: projectId },
            query: { prompt_mode: promptMode },
        },
    });

    return data.models;
};
