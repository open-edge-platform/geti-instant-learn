/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext } from 'react';

import { useLoadImageQuery } from '../hooks/use-load-image-query.hook';
import type { RegionOfInterest } from '../types';

type AnnotatorContext = {
    // Media items
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
