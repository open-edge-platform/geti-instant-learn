/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, use, useEffect, useState } from 'react';

import { LabelType, VisualPromptType } from '@geti-prompt/api';

import { useGetPrompt } from './api/use-get-prompt';
import { useProjectLabels } from './use-project-labels.hook';
import { usePromptIdFromUrl } from './use-prompt-id-from-url';

interface VisualPromptContextProps {
    promptId: string | null;
    setPromptId: (id: string | null) => void;
    prompt: VisualPromptType | undefined;

    selectedLabelId: string | null;
    setSelectedLabelId: (id: string) => void;
    selectedLabel: LabelType | null;
    labels: LabelType[];
}

const VisualPromptContext = createContext<VisualPromptContextProps | null>(null);

interface VisualPromptProviderProps {
    children: ReactNode;
}

const useSelectedLabel = (prompt: VisualPromptType | undefined) => {
    const labels = useProjectLabels();

    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(null);

    const selectedLabel: LabelType | null = labels.find(({ id }) => id === selectedLabelId) ?? null;

    // Auto-select label
    useEffect(() => {
        if (prompt?.annotations !== undefined) {
            const labelId = prompt.annotations[0].label_id;

            labelId && setSelectedLabelId(labelId);

            return;
        }

        if (labels.length > 0) {
            setSelectedLabelId(labels[0].id);
        }
    }, [labels, prompt?.annotations]);

    return {
        selectedLabel,
        selectedLabelId,
        setSelectedLabelId,
        labels,
    };
};

export const VisualPromptProvider = ({ children }: VisualPromptProviderProps) => {
    const { promptId, setPromptId } = usePromptIdFromUrl();
    const prompt = useGetPrompt(promptId);

    const { selectedLabel, selectedLabelId, setSelectedLabelId, labels } = useSelectedLabel(prompt);

    return (
        <VisualPromptContext
            value={{
                promptId,
                setPromptId,
                prompt,

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
