/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useRef } from 'react';

import { $api, FramesResponseType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { uniqBy } from 'lodash-es';

const LIMIT = 30;
const INITIAL_OFFSET = 0;

const ACTIVE_FRAME_OFFSET = Math.floor(LIMIT / 2);

const useFramesQuery = (sourceId: string, activeFrameIdx: number) => {
    const { projectId } = useProjectIdentifier();
    const activeFrameIdxRef = useRef(activeFrameIdx);

    const queryOffset = Math.max(activeFrameIdxRef.current - ACTIVE_FRAME_OFFSET, INITIAL_OFFSET);

    return $api.useInfiniteQuery(
        'get',
        '/api/v1/projects/{project_id}/sources/{source_id}/frames',
        {
            params: {
                path: { project_id: projectId, source_id: sourceId },
                query: {
                    limit: LIMIT,
                },
            },
        },
        {
            pageParamName: 'offset',
            initialPageParam: queryOffset,
            getNextPageParam: ({ pagination }: FramesResponseType) => {
                const { offset, limit, total } = pagination;
                const nextPage = offset + limit;

                return nextPage < total ? nextPage : undefined;
            },
            getPreviousPageParam: ({ pagination }: FramesResponseType) => {
                const { offset, limit } = pagination;

                if (offset === 0) {
                    return undefined;
                }

                const previousPage = Math.max(0, offset - limit);

                return previousPage;
            },
        }
    );
};

export const useGetFrames = (sourceId: string, activeFrameIdx: number) => {
    const {
        data,
        hasNextPage,
        isFetchingNextPage,
        isPending,
        fetchNextPage,
        isFetchingPreviousPage,
        fetchPreviousPage,
        hasPreviousPage,
    } = useFramesQuery(sourceId, activeFrameIdx);

    const frames = uniqBy(data?.pages.flatMap((page) => page.frames) ?? [], (frame) => frame.index);

    const handleFetchNextPage = async () => {
        if (hasNextPage && !isFetchingNextPage) {
            await fetchNextPage();
        }
    };

    const handleFetchPreviousPage = async () => {
        if (hasPreviousPage && !isFetchingPreviousPage) {
            await fetchPreviousPage();
        }
    };

    const framesCount = data?.pages?.at(0)?.pagination?.total ?? 0;

    return {
        frames,
        fetchNextPage: handleFetchNextPage,
        isFetchingNextFrames: isFetchingNextPage,
        fetchPreviousPage: handleFetchPreviousPage,
        isPending,
        framesCount,
    } as const;
};
