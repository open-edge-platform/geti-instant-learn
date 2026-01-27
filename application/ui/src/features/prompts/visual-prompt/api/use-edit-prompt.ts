/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, VisualPromptType } from '@/api';
import { useProjectIdentifier } from '@/hooks';

import { convertAnnotationsToDTO } from '../../../../shared/utils';
import { useAnnotationActions } from '../../../annotator/providers/annotation-actions-provider.component';

const useEditPromptMutation = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('put', '/api/v1/projects/{project_id}/prompts/{prompt_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/prompts', { params: { path: { project_id: projectId } } }],
            ],
        },
    });
};

export const useEditPrompt = () => {
    const { projectId } = useProjectIdentifier();
    const { annotations } = useAnnotationActions();
    const editPromptMutation = useEditPromptMutation();

    const editPrompt = (prompt: VisualPromptType) => {
        editPromptMutation.mutate({
            body: {
                type: prompt.type,
                frame_id: prompt.frame_id,
                annotations: convertAnnotationsToDTO(annotations),
            },
            params: {
                path: {
                    project_id: projectId,
                    prompt_id: prompt.id,
                },
            },
        });
    };

    return {
        mutate: editPrompt,
        isPending: editPromptMutation.isPending,
    };
};
