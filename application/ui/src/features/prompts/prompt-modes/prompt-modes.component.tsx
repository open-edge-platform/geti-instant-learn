/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from 'react';

import { usePromptMode } from '@geti-prompt/hooks';
import { Flex, Text, ToggleButtons } from '@geti/ui';
import { useSearchParams } from 'react-router-dom';

import styles from './prompt-modes.module.scss';

type PromptMode = 'visual' | 'text';

const VISUAL_PROMPT_MODE = 'Visual Prompt';
const TEXT_PROMPT_MODE = 'Text Prompt';

const OPTIONS = [VISUAL_PROMPT_MODE, TEXT_PROMPT_MODE];

const getSelectedUIPromptMode = (mode: PromptMode = 'visual') => {
    if (mode === 'visual') {
        return VISUAL_PROMPT_MODE;
    }
    return TEXT_PROMPT_MODE;
};

const getSelectedPromptMode = (mode: string): PromptMode => {
    if (mode === VISUAL_PROMPT_MODE) {
        return 'visual';
    }
    return 'text';
};

export const PromptModes = () => {
    const [searchParams, setSearchParams] = useSearchParams();
    const mode = usePromptMode();

    const selectedMode = getSelectedUIPromptMode(mode);

    const handleModeChange = (option: string) => {
        const newMode = getSelectedPromptMode(option);
        searchParams.set('mode', newMode);
        setSearchParams(searchParams);
    };

    useEffect(() => {
        const localMode = searchParams.get('mode');

        if (localMode === null) {
            searchParams.set('mode', 'visual');
            setSearchParams(searchParams);
        }
    }, [searchParams, setSearchParams]);

    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Text UNSAFE_className={styles.label}>Prompt Mode</Text>
            <ToggleButtons options={OPTIONS} selectedOption={selectedMode} onOptionChange={handleModeChange} />
        </Flex>
    );
};
