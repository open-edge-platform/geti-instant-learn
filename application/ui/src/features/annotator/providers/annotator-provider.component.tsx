/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext } from 'react';

import { useProjectIdentifier } from '@geti-prompt/hooks';
import { useSuspenseQuery, UseSuspenseQueryResult } from '@tanstack/react-query';

import { getImageData, loadImage } from '../tools/utils';
import type { RegionOfInterest } from '../types';

const useLoadImageQuery = (frameId: string): UseSuspenseQueryResult<ImageData, unknown> => {
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

type AnnotatorContext = {
    roi: RegionOfInterest;
    frameId: string;
    image: ImageData;
};

const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({ frameId, children }: { frameId: string; children: ReactNode }) => {
    const imageQuery = useLoadImageQuery(frameId);

    return (
        <AnnotatorProviderContext
            value={{
                image: imageQuery.data,
                frameId,
                roi: { x: 0, y: 0, width: imageQuery.data.width, height: imageQuery.data.height },
            }}
        >
            {children}
        </AnnotatorProviderContext>
    );
};

export const useAnnotator = () => {
    const context = useContext(AnnotatorProviderContext);

    if (context === null) {
        throw new Error('useAnnotator was used outside of AnnotatorProvider');
    }

    return context;
};
