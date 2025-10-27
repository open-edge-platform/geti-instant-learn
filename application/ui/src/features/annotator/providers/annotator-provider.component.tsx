/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext, useState, type Dispatch, type SetStateAction } from 'react';

import { useLoadImageQuery } from '../hooks/use-load-image-query.hook';
import { ToolType } from '../tools/interface';
import type { RegionOfInterest } from '../types';

type AnnotatorContext = {
    // Tools
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType | null>>;

    // Media item
    roi: RegionOfInterest;
    frameId: string;
    image: ImageData;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({ frameId, children }: { frameId: string; children: ReactNode }) => {
    const [activeTool, setActiveTool] = useState<ToolType | null>(null);

    const imageQuery = useLoadImageQuery(frameId);

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,

                image: imageQuery.data,
                frameId,
                roi: { x: 0, y: 0, width: imageQuery.data.width, height: imageQuery.data.height },
            }}
        >
            {children}
        </AnnotatorProviderContext.Provider>
    );
};

export const useAnnotator = () => {
    const context = useContext(AnnotatorProviderContext);

    if (context === null) {
        throw new Error('useAnnotator was used outside of AnnotatorProvider');
    }

    return context;
};
