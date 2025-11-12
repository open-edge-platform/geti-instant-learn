/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, VisualPromptType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

import { useAnnotationActions } from '../../../annotator/providers/annotation-actions-provider.component';
import { useAnnotator } from '../../../annotator/providers/annotator-provider.component';
import { convertAnnotationsToDTO } from './utils';

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
    const { roi } = useAnnotator();
    const { annotations } = useAnnotationActions();
    const editPromptMutation = useEditPromptMutation();

    const editPrompt = (prompt: VisualPromptType) => {
        editPromptMutation.mutate({
            body: {
                type: prompt.type,
                frame_id: prompt.frame_id,
                annotations: convertAnnotationsToDTO(annotations, roi),
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
