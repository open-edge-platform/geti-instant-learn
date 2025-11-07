/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, use, useEffect, useState } from 'react';

import { LabelType } from '@geti-prompt/api';
import { useProjectLabels } from '@geti-prompt/hooks';

interface VisualPromptContextProps {
    promptId: string | null;
    setPromptId: (id: string) => void;

    selectedLabelId: string;
    setSelectedLabelId: (id: string) => void;
    selectedLabel: LabelType;
    labels: LabelType[];
}

const VisualPromptContext = createContext<VisualPromptContextProps | null>(null);

interface VisualPromptProviderProps {
    children: ReactNode;
}

const PLACEHOLDER_LABEL: LabelType = { id: 'placeholder', name: 'No label', color: 'var(--annotation-fill)' };

export const VisualPromptProvider = ({ children }: VisualPromptProviderProps) => {
    const labels = useProjectLabels();
    const [selectedLabelId, setSelectedLabelId] = useState<string>(PLACEHOLDER_LABEL.id);
    const [promptId, setPromptId] = useState<string | null>(null);

    const selectedLabel: LabelType = labels.find(({ id }) => id === selectedLabelId) ?? PLACEHOLDER_LABEL;

    useEffect(() => {
        if (labels.length > 0) {
            setSelectedLabelId(labels[0].id);
        }
    }, [labels]);

    return (
        <VisualPromptContext
            value={{
                promptId,
                setPromptId,

                setSelectedLabelId,
                selectedLabelId,
                selectedLabel,
                labels,
            }}
        >
            {children}
        </VisualPromptContext>
    );
};

export const useVisualPrompt = () => {
    const context = use(VisualPromptContext);

    if (context === null) {
        throw new Error('useVisualPrompt must be used within a VisualPromptProvider');
    }

    return context;
};
