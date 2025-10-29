/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, use, useState } from 'react';

import { ActionButton } from '@geti/ui';
import { Collapse, Expand } from '@geti/ui/icons';

const FullScreenModeContext = createContext<{
    isFullScreenMode: boolean;
    setIsFullScreenMode: (isFullScreenMode: boolean) => void;
} | null>(null);

export const FullScreenModeProvider = ({ children }: { children: ReactNode }) => {
    const [isFullScreenMode, setIsFullScreenMode] = useState<boolean>(false);

    return <FullScreenModeContext value={{ isFullScreenMode, setIsFullScreenMode }}>{children}</FullScreenModeContext>;
};

export const useFullScreenMode = () => {
    const context = use(FullScreenModeContext);

    if (context === null) {
        throw new Error('useFullScreenMode must be used within a FullScreenModeProvider');
    }

    return context;
};

export const FullScreenMode = () => {
    const { isFullScreenMode, setIsFullScreenMode } = useFullScreenMode();

    return (
        <ActionButton
            isQuiet
            aria-label={isFullScreenMode ? 'Close full screen' : 'Open full screen'}
            onPress={() => setIsFullScreenMode(!isFullScreenMode)}
        >
            {isFullScreenMode ? <Collapse /> : <Expand />}
        </ActionButton>
    );
};
