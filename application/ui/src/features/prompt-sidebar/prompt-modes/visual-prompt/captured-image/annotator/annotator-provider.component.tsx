/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { CapturedImageType } from '../../types';
import { ToolType } from './tool.interface';
import { RegionOfInterest } from './types';

type AnnotatorContext = {
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType>>;
    size: { width: number; height: number };
    roi: RegionOfInterest;
    image: CapturedImageType;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({
    children,
    size,
    image,
}: {
    children: ReactNode;
    size: { width: number; height: number };
    image: CapturedImageType;
}) => {
    const [activeTool, setActiveTool] = useState<ToolType>('bounding-box');

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,
                size,
                image,
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
