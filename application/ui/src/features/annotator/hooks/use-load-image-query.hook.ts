/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-prompt/hooks';
import { useSuspenseQuery, UseSuspenseQueryResult } from '@tanstack/react-query';

import { getImageData, loadImage } from '../tools/utils';

export const useLoadImageQuery = (frameId: string): UseSuspenseQueryResult<ImageData, unknown> => {
    const { projectId } = useProjectIdentifier();

    return useSuspenseQuery({
        queryKey: ['mediaItem', frameId, projectId],
        queryFn: async () => {
            if (frameId === undefined) {
                throw new Error("Can't fetch undefined media item");
            }

            const imageUrl = `${import.meta.env.PUBLIC_API_URL}/api/v1/projects/${projectId}/frames/${frameId}`;
            const image = await loadImage(imageUrl);

            return getImageData(image);
        },
        // The image of a media item never changes so we don't want to refetch stale data
        staleTime: Infinity,
    });
};
