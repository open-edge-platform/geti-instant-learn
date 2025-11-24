/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

export const useActivateFrameMutation = () => {
    return $api.useMutation('post', '/api/v1/projects/{project_id}/sources/{source_id}/frames/{index}');
};

export const useActivateFrame = () => {
    const { projectId } = useProjectIdentifier();
    const activateFrameMutation = useActivateFrameMutation();

    const activateFrame = ({
        sourceId,
        index,
        onSuccess,
    }: {
        sourceId: string;
        index: number;
        onSuccess: () => void;
    }) => {
        activateFrameMutation.mutate(
            {
                params: {
                    path: {
                        project_id: projectId,
                        source_id: sourceId,
                        index,
                    },
                },
            },
            {
                onSuccess,
            }
        );
    };

    return {
        mutate: activateFrame,
        isPending: activateFrameMutation.isPending,
    };
};
