/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { getQueryKey } from '@/query-client';
import { useQueryClient } from '@tanstack/react-query';

export const useUpdateProjectDevice = () => {
    const { projectId } = useProjectIdentifier();
    const queryClient = useQueryClient();

    // TODO: Replace with proper data type once the API is ready.
    const mutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            error: { notify: true },
        },
    });

    const updateDevice = (device: string): void => {
        mutation.mutate(
            {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                body: { device } as any,
                params: { path: { project_id: projectId } },
            },
            {
                onSuccess: async () => {
                    await queryClient.invalidateQueries({
                        queryKey: getQueryKey([
                            'get',
                            '/api/v1/projects/{project_id}',
                            { params: { path: { project_id: projectId } } },
                        ]),
                    });
                },
            }
        );
    };

    return {
        mutate: updateDevice,
        isPending: mutation.isPending,
    };
};
