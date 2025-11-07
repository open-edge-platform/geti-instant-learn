/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, useContext, useEffect, useState, type Dispatch, type SetStateAction } from 'react';

import { LabelType } from '@geti-prompt/api';
import { useProjectLabels } from '@geti-prompt/hooks';

import { useLoadImageQuery } from '../hooks/use-load-image-query.hook';
import type { RegionOfInterest } from '../types';

type AnnotatorContext = {
    // Media items
    roi: RegionOfInterest;
    frameId: string;
    image: ImageData;

    // Labels
    selectedLabel: LabelType;
    selectedLabelId: string;
    setSelectedLabelId: Dispatch<SetStateAction<string>>;
};

const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

const PLACEHOLDER_LABEL: LabelType = { id: 'placeholder', name: 'No label', color: 'var(--annotation-fill)' };

export const AnnotatorProvider = ({ frameId, children }: { frameId: string; children: ReactNode }) => {
    const labels = useProjectLabels();
    const [selectedLabelId, setSelectedLabelId] = useState<string>(PLACEHOLDER_LABEL.id);

    const imageQuery = useLoadImageQuery(frameId);
    const selectedLabel = labels.find(({ id }) => id === selectedLabelId) || PLACEHOLDER_LABEL;

    useEffect(() => {
        if (labels.length > 0) {
            setSelectedLabelId(labels[0].id);
        }
    }, [labels]);

    return (
        <AnnotatorProviderContext
            value={{
                setSelectedLabelId,
                selectedLabelId,
                selectedLabel,

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
