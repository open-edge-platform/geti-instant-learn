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
                    page: 0,
                    page_size: 20,
                },
            },
        },
        {
            pageParamName: 'page',
            initialPageParam: 0,
            getNextPageParam: ({ page, page_size, total }: FramesResponseType) => {
                const nextPage = page + 1;

                return page_size * page < total ? nextPage : undefined;
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

    return {
        frames,
        loadMoreFrames,
        isFetchingNextFrames: isFetchingNextPage,
        isPending,
    } as const;
};
