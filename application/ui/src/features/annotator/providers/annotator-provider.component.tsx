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
    selectedLabelId: string;
    setSelectedLabelId: Dispatch<SetStateAction<string>>;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const PLACEHOLDER_LABEL: LabelType = { id: 'placeholder', name: 'No label', color: 'var(--annotation-fill)' };

export const AnnotatorProvider = ({ frameId, children }: { frameId: string; children: ReactNode }) => {
    const labels = useProjectLabels();
    const [activeTool, setActiveTool] = useState<ToolType | null>(null);
    const [selectedLabelId, setSelectedLabelId] = useState<string>(PLACEHOLDER_LABEL.id);

    const imageQuery = useLoadImageQuery(frameId);
    const selectedLabel = labels.find(({ id }) => id === selectedLabelId) || PLACEHOLDER_LABEL;

    useEffect(() => {
        if (labels.length > 0) {
            setSelectedLabelId(labels[0].id);
        }
    }, [labels]);

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,

                setSelectedLabelId,
                selectedLabelId,
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
