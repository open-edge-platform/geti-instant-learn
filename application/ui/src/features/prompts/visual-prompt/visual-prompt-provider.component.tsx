/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, use, useEffect, useState } from 'react';

import { LabelType, VisualPromptType } from '@geti-prompt/api';
import { useProjectLabels } from '@geti-prompt/hooks';

import { useGetPrompt } from './api/use-get-prompt';

interface VisualPromptContextProps {
    promptId: string | null;
    setPromptId: (id: string) => void;
    prompt: VisualPromptType | undefined;

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
    const prompt = useGetPrompt(promptId);

    const selectedLabel: LabelType = labels.find(({ id }) => id === selectedLabelId) ?? PLACEHOLDER_LABEL;

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
