/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { usePromptMode } from './prompt-modes.component';
import { TextPrompt } from './text-prompt/text-prompt.component';
import { VisualPrompt } from './visual-prompt/visual-prompt.component';

export const PromptMode = () => {
    const mode = usePromptMode();

    if (mode === 'visual') {
        return <VisualPrompt />;
    }

    return <TextPrompt />;
};
