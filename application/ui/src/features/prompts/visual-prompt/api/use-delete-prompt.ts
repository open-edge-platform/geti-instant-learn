/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { toast } from '@geti/ui';

import { useSelectedFrame } from '../../../../shared/selected-frame-provider.component';
import { useVisualPrompt } from '../visual-prompt-provider.component';

export const useDeletePrompt = () => {
    const { projectId } = useProjectIdentifier();
    const { prompt, setPromptId } = useVisualPrompt();
    const { selectedFrameId, setSelectedFrameId } = useSelectedFrame();

    return $api.useMutation('delete', '/api/v1/projects/{project_id}/prompts/{prompt_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/prompts', { params: { path: { project_id: projectId } } }],
            ],
        },
        onSuccess: () => {
            if (selectedFrameId === prompt?.frame_id) {
                setSelectedFrameId(null);
            }

            setPromptId(null);
            toast({
                type: 'success',
                message: 'Prompt deleted successfully.',
            });
        },
    });
};
