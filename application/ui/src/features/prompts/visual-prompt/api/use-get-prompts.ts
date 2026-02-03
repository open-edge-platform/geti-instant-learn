/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, VisualPromptListType } from '@/api';
import { useProjectIdentifier } from '@/hooks';

export const useGetPrompts = (): VisualPromptListType['prompts'] => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/prompts', {
        params: { path: { project_id: projectId } },
    });

    return data.prompts ?? [];
};
