/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect } from 'react';

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

    const handleModeChange = useCallback(
        (option: string) => {
            const newMode = getSelectedPromptMode(option);

            setSearchParams((previousSearchParams) => {
                previousSearchParams.set('mode', newMode);

                return previousSearchParams;
            });
        },
        [setSearchParams]
    );

    const mode = searchParams.get('mode');

    useEffect(() => {
        if (mode === null) {
            handleModeChange('visual');
        }
    }, [mode, handleModeChange]);

    return [(mode ?? 'visual') as PromptMode, handleModeChange] as const;
};
