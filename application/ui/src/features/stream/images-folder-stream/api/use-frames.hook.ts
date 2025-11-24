/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, FramesResponseType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';

const useFramesQuery = (sourceId: string) => {
    const { projectId } = useProjectIdentifier();

    return $api.useInfiniteQuery(
        'get',
        '/api/v1/projects/{project_id}/sources/{source_id}/frames',
        {
            params: {
                path: { project_id: projectId, source_id: sourceId },
                query: {
                    offset: 0,
                    limit: 30,
                },
            },
        },
        {
            pageParamName: 'offset',
            initialPageParam: 0,
            getNextPageParam: ({ pagination }: FramesResponseType) => {
                const { offset, limit, total } = pagination;
                const nextPage = offset + limit;

                return nextPage < total ? nextPage : undefined;
            },
        }
    );
};

export const useGetFrames = (sourceId: string) => {
    const { data, hasNextPage, isFetchingNextPage, isPending, fetchNextPage } = useFramesQuery(sourceId);

    const frames = data?.pages.flatMap((page) => page.frames) ?? [];

    const loadMoreFrames = async () => {
        if (hasNextPage && !isFetchingNextPage) {
            await fetchNextPage();
        }
    };

    const framesCount = data?.pages?.at(0)?.pagination?.total ?? 0;

    return {
        frames,
        loadMoreFrames,
        isFetchingNextFrames: isFetchingNextPage,
        isPending,
        framesCount,
    } as const;
};
