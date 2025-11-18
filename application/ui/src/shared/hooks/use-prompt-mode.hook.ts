/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useSearchParams } from 'react-router-dom';

export type PromptMode = 'visual' | 'text';

const getSelectedPromptMode = (mode: string): PromptMode => {
    if (mode.toLocaleLowerCase().includes('visual')) {
        return 'visual';
    }
    return 'text';
};

export const usePromptMode = (): [PromptMode, (mode: string) => void] => {
    const [searchParams, setSearchParams] = useSearchParams();

    const mode = (searchParams.get('mode') as PromptMode) ?? 'visual';

    const handleModeChange = (option: string) => {
        const newMode = getSelectedPromptMode(option);
        const newSearchParams = new URLSearchParams(searchParams);

        newSearchParams.set('mode', newMode);
        setSearchParams(newSearchParams);
    };

    return [mode, handleModeChange] as const;
};
