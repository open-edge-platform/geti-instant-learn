/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext, useEffect, useState, type Dispatch, type SetStateAction } from 'react';

import { LabelType } from '@geti-prompt/api';
import { useProjectLabels } from '@geti-prompt/hooks';

import { useLoadImageQuery } from '../hooks/use-load-image-query.hook';
import { ToolType } from '../tools/interface';
import type { RegionOfInterest } from '../types';

type AnnotatorContext = {
    // Tools
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType | null>>;

    // Media items
    roi: RegionOfInterest;
    frameId: string;
    image: ImageData;

    // Labels
    selectedLabel: LabelType;
    setSelectedLabel: Dispatch<SetStateAction<LabelType>>;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

const PLACEHOLDER_LABEL = { id: 'placeholder', name: 'No label', color: 'var(--annotation-fill)', isPrediction: false };

export const AnnotatorProvider = ({ frameId, children }: { frameId: string; children: ReactNode }) => {
    const labels = useProjectLabels();
    const [activeTool, setActiveTool] = useState<ToolType | null>(null);
    const [selectedLabel, setSelectedLabel] = useState<LabelType>(PLACEHOLDER_LABEL);

    const imageQuery = useLoadImageQuery(frameId);

    useEffect(() => {
        if (labels.length > 0 && selectedLabel.id === PLACEHOLDER_LABEL.id) {
            setSelectedLabel(labels[0]);
        }
    }, [labels, selectedLabel.id]);

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,

                setSelectedLabel,
                selectedLabel,

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
