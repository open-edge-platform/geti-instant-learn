/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { toast } from '@geti/ui';

import { useVisualPrompt } from '../visual-prompt-provider.component';

export const useDeletePrompt = () => {
    const { projectId } = useProjectIdentifier();
    const { setPromptId } = useVisualPrompt();

    return $api.useMutation('delete', '/api/v1/projects/{project_id}/prompts/{prompt_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/prompts', { params: { path: { project_id: projectId } } }],
            ],
        },
        onSuccess: () => {
            setPromptId(null);
            toast({
                type: 'success',
                message: 'Prompt deleted successfully.',
            });
        },
    });
};
