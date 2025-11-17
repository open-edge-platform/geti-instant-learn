/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, use, useEffect, useState } from 'react';

import { LabelType, VisualPromptType } from '@geti-prompt/api';

import { useSelectedFrame } from '../../../shared/selected-frame-provider.component';
import { useGetPrompt } from './api/use-get-prompt';
import { useProjectLabels } from './use-project-labels.hook';
import { usePromptIdFromUrl } from './use-prompt-id-from-url';

interface VisualPromptContextProps {
    promptId: string | null;
    setPromptId: (id: string | null) => void;
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
    const selectedLabel: LabelType = labels.find(({ id }) => id === selectedLabelId) ?? PLACEHOLDER_LABEL;

    const { promptId, setPromptId } = usePromptIdFromUrl();
    const { selectedFrameId, setSelectedFrameId } = useSelectedFrame();
    const prompt = useGetPrompt(promptId);

    // Auto-load frame
    useEffect(() => {
        if (prompt?.frame_id && selectedFrameId !== prompt.frame_id) {
            setSelectedFrameId(prompt.frame_id);
        }
    }, [prompt?.frame_id, selectedFrameId, setSelectedFrameId]);

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
