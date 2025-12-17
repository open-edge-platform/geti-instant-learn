/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { toast } from '@geti/ui';
import { v4 as uuid } from 'uuid';

import { convertAnnotationsToDTO } from '../../../../shared/utils';
import { useFullScreenMode } from '../../../annotator/actions/full-screen-mode.component';
import { useAnnotationActions } from '../../../annotator/providers/annotation-actions-provider.component';
import { useAnnotator } from '../../../annotator/providers/annotator-provider.component';
import { useVisualPrompt } from '../visual-prompt-provider.component';
import { useEditPrompt } from './use-edit-prompt';

const useSavePromptMutation = () => {
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
    });
};

export const useSavePrompt = () => {
    const { projectId } = useProjectIdentifier();
    const { annotations } = useAnnotationActions();
    const { roi, frameId } = useAnnotator();
    const { setPromptId, prompt } = useVisualPrompt();
    const { setIsFullScreenMode } = useFullScreenMode();

    const savePromptMutation = useSavePromptMutation();
    const editPromptMutation = useEditPrompt();

    const createPrompt = () => {
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
                    setIsFullScreenMode(false);
                    setPromptId(id);
                    toast({
                        type: 'success',
                        message: 'Prompt created successfully.',
                    });
                },
            }
        );
    };

    const savePrompt = () => {
        if (prompt === undefined) {
            createPrompt();
        } else {
            editPromptMutation.mutate(prompt);
        }
    };

    return {
        mutate: savePrompt,
        isPending: savePromptMutation.isPending || editPromptMutation.isPending,
    };
};
