/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { v4 as uuid } from 'uuid';

import { useAnnotationActions } from '../../../annotator/providers/annotation-actions-provider.component';
import { useAnnotator } from '../../../annotator/providers/annotator-provider.component';
import { useVisualPrompt } from '../visual-prompt-provider.component';
import { convertAnnotationsToDTO } from './utils';

const useSavePromptMutation = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('post', '/api/v1/projects/{project_id}/prompts', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/prompts', { params: { path: { project_id: projectId } } }],
            ],
        },
    });
};

export const useSavePrompt = () => {
    const { projectId } = useProjectIdentifier();
    const { annotations } = useAnnotationActions();
    const { roi } = useAnnotator();
    const { setPromptId } = useVisualPrompt();

    const savePromptMutation = useSavePromptMutation();

    const savePrompt = (frameId: string) => {
        savePromptMutation.mutate(
            {
                body: {
                    id: uuid(),
                    type: 'VISUAL',
                    frame_id: frameId,
                    annotations: convertAnnotationsToDTO(annotations, roi),
                },
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            },
            {
                onSuccess: ({ id }) => {
                    setPromptId(id);
                },
            }
        );
    };

    return {
        mutate: savePrompt,
        isPending: savePromptMutation.isPending,
    };
};
