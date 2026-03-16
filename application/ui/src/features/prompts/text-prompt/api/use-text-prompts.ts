/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, TextPromptType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { toast } from '@geti/ui';

export const useGetTextPrompts = (): TextPromptType[] => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/prompts', {
        params: { path: { project_id: projectId } },
    });

    return (data.prompts ?? []).filter((prompt): prompt is TextPromptType => prompt.type === 'TEXT');
};

export const useCreateTextPrompt = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('post', '/api/v1/projects/{project_id}/prompts', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/prompts', { params: { path: { project_id: projectId } } }],
            ],
            error: {
                notify: true,
            },
        },
        onSuccess: () => {
            toast({
                type: 'success',
                message: 'Prompt created successfully.',
            });
        },
    });
};

export const useDeleteTextPrompt = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('delete', '/api/v1/projects/{project_id}/prompts/{prompt_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/prompts', { params: { path: { project_id: projectId } } }],
            ],
            error: {
                notify: true,
            },
        },
        onSuccess: () => {
            toast({
                type: 'success',
                message: 'Prompt deleted successfully.',
            });
        },
    });
};
