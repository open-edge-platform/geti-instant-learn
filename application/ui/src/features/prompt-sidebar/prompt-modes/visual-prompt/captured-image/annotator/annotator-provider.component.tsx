/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { ToolType } from './tool.interface';
import { RegionOfInterest } from './types';

type AnnotatorContext = {
    // Tools
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType>>;

    roi: RegionOfInterest;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({ children }: { children: ReactNode }) => {
    const [activeTool, setActiveTool] = useState<ToolType>('bounding-box');

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,
                roi: { x: 0, y: 0, width: 500, height: 500 },
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
