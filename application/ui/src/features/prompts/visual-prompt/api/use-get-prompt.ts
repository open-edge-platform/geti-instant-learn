/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, VisualPromptType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

export const useGetPrompt = (promptId: string | null) => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery(
        'get',
        '/api/v1/projects/{project_id}/prompts/{prompt_id}',
        {
            params: {
                path: {
                    project_id: projectId,
                    prompt_id: String(promptId),
                },
            },
        },
        {
            enabled: promptId !== null,
        }
    );

    // In this place we're sure we only get a VisualPromptType
    return data as VisualPromptType | undefined;
};
