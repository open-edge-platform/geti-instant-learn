/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useSelectedFrame } from '../../stream/selected-frame-provider.component';
import { TextPrompt } from '../text-prompt/text-prompt.component';
import { VisualPromptProvider } from '../visual-prompt/visual-prompt-provider.component';
import { VisualPrompt } from '../visual-prompt/visual-prompt.component';
import { usePromptMode } from './prompt-modes.component';

export const PromptMode = () => {
    const mode = usePromptMode();
    const { selectedFrameId } = useSelectedFrame();

    if (mode === 'visual') {
        return (
            // When user captures a new frame, we don't want to store previous prompts state in the VisualPromptProvider
            // For each frame we want to have a new instance of the VisualPromptProvider
            <VisualPromptProvider key={selectedFrameId}>
                <VisualPrompt />
            </VisualPromptProvider>
        );
    }

    return <TextPrompt />;
};
