/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Button } from '@geti/ui';

import { useSelectedFrame } from '../../shared/selected-frame-provider.component';

const useCaptureFrameMutation = () => {
    return $api.useMutation('post', '/api/v1/projects/{project_id}/frames');
};

const useCaptureFrame = () => {
    const { projectId } = useProjectIdentifier();
    const captureFrameMutation = useCaptureFrameMutation();
    const { setSelectedFrameId } = useSelectedFrame();

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
                    setSelectedFrameId(frame_id);
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
