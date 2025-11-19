/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, ReactNode, use, useState } from 'react';

interface SelectedFrameContextProps {
    selectedFrameId: string | null;
    setSelectedFrameId: (id: string | null) => void;
}

const SelectedFrameContext = createContext<SelectedFrameContextProps | null>(null);

interface SelectedFrameProviderProps {
    children: ReactNode;
}

export const SelectedFrameProvider = ({ children }: SelectedFrameProviderProps) => {
    const [selectedFrameId, setSelectedFrameId] = useState<string | null>(null);

    return <SelectedFrameContext value={{ selectedFrameId, setSelectedFrameId }}>{children}</SelectedFrameContext>;
};

export const useSelectedFrame = (): SelectedFrameContextProps => {
    const context = use(SelectedFrameContext);

    if (context === null) {
        throw new Error('useSelectedFrame must be used within a SelectedFrameProvider');
    }

    return context;
};
