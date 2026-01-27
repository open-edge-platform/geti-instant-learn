/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from 'react';

import { $api, VisualPromptType } from '@/api';
import { useProjectIdentifier } from '@/hooks';

import { useSelectedFrame } from '../../../../shared/selected-frame-provider.component';

const useGetPromptQuery = (promptId: string | null) => {
    const { projectId } = useProjectIdentifier();
    return $api.useQuery(
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
};

export const useGetPrompt = (promptId: string | null) => {
    const { setSelectedFrameId } = useSelectedFrame();
    const { data, isSuccess } = useGetPromptQuery(promptId);

    // In this place we're sure we only get a VisualPromptType
    const prompt = data as VisualPromptType | undefined;

    useEffect(() => {
        if (isSuccess && prompt?.frame_id !== undefined) {
            setSelectedFrameId(prompt.frame_id);
        }
    }, [prompt?.frame_id, isSuccess, setSelectedFrameId]);

    return prompt;
};
