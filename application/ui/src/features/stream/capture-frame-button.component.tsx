/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Button } from '@geti/ui';

import { useSelectedFrame } from '../../shared/selected-frame-provider.component';
import { usePromptIdFromUrl } from '../prompts/visual-prompt/use-prompt-id-from-url';

const useCaptureFrameMutation = () => {
    return $api.useMutation('post', '/api/v1/projects/{project_id}/frames');
};

const useCaptureFrame = () => {
    const { projectId } = useProjectIdentifier();
    const captureFrameMutation = useCaptureFrameMutation();
    const { setSelectedFrameId } = useSelectedFrame();
    const { setPromptId } = usePromptIdFromUrl();

    const changeSelectedFrame = (id: string | null) => {
        setPromptId(null);
        setSelectedFrameId(id);
    };

    const captureFrame = async () => {
        captureFrameMutation.mutate(
            {
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            },
            {
                onSuccess: ({ frame_id }) => {
                    changeSelectedFrame(frame_id);
                },
            }
        );
    };

    return {
        captureFrame,
        isPending: captureFrameMutation.isPending,
    };
};

export const CaptureFrameButton = () => {
    const { captureFrame, isPending } = useCaptureFrame();

    return (
        <Button variant={'primary'} staticColor={'white'} style={'fill'} onPress={captureFrame} isPending={isPending}>
            Capture
        </Button>
    );
};
