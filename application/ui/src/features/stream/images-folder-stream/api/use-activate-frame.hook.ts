/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { getQueryKey } from '@/query-client';
import { useQueryClient } from '@tanstack/react-query';

export const useActivateFrameMutation = () => {
    return $api.useMutation('post', '/api/v1/projects/{project_id}/sources/{source_id}/frames/{index}');
};

export const useActivateFrame = () => {
    const queryClient = useQueryClient();
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
        const params = {
            path: {
                project_id: projectId,
                source_id: sourceId,
                index,
            },
        };

        activateFrameMutation.mutate(
            {
                params,
            },
            {
                onSuccess: async () => {
                    await queryClient.invalidateQueries({
                        queryKey: getQueryKey([
                            'get',
                            '/api/v1/projects/{project_id}/sources/{source_id}/frames/index',
                            {
                                params: {
                                    path: {
                                        project_id: projectId,
                                        source_id: sourceId,
                                    },
                                },
                            },
                        ]),
                    });
                    onSuccess();
                },
            }
        );
    };

    return {
        mutate: activateFrame,
        isPending: activateFrameMutation.isPending,
    };
};
