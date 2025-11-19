/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VisualPromptProvider } from '../visual-prompt/visual-prompt-provider.component';
import { VisualPrompt } from '../visual-prompt/visual-prompt.component';

export const PromptMode = () => {
    return (
        <VisualPromptProvider>
            <VisualPrompt />
        </VisualPromptProvider>
    );

    /* TODO: Uncomment when we support text prompt
    const [mode] = usePromptMode();

    if (mode === 'visual') {
        return (
            <VisualPromptProvider>
                <VisualPrompt />
            </VisualPromptProvider>
        );
    }

    return <TextPrompt />;*/
};
