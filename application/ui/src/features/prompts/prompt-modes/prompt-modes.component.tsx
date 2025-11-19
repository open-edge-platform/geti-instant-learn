/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from 'react';

import { usePromptMode, type PromptMode } from '@geti-prompt/hooks';
import { Flex, Text, ToggleButtons } from '@geti/ui';
import { useSearchParams } from 'react-router-dom';

import styles from './prompt-modes.module.scss';

const VISUAL_PROMPT_MODE = 'Visual Prompt';
const TEXT_PROMPT_MODE = 'Text Prompt';

const OPTIONS = [VISUAL_PROMPT_MODE, TEXT_PROMPT_MODE];

const getSelectedUIPromptMode = (mode: PromptMode = 'visual') => {
    if (mode === 'visual') {
        return VISUAL_PROMPT_MODE;
    }
    return TEXT_PROMPT_MODE;
};

export const PromptModes = () => {
    const [searchParams] = useSearchParams();
    const [mode, setPromptMode] = usePromptMode();
    const localMode = searchParams.get('mode');

    const selectedMode = getSelectedUIPromptMode(mode);

    useEffect(() => {
        if (localMode === null) {
            setPromptMode('visual');
        }
    }, [localMode, setPromptMode]);

    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Text UNSAFE_className={styles.label}>Prompt Mode</Text>
            <ToggleButtons options={OPTIONS} selectedOption={selectedMode} onOptionChange={setPromptMode} />
        </Flex>
    );
};
