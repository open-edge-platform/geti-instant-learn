/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { usePromptMode, type PromptMode } from '@/hooks';
import { Flex, Text, ToggleButtons } from '@geti/ui';

import styles from './prompt-modes.module.scss';

const OPTIONS: PromptMode[] = ['VISUAL', 'TEXT'];

const LABELS: Record<PromptMode, string> = {
    VISUAL: 'Visual Prompt',
    TEXT: 'Text Prompt',
};

export const PromptModes = () => {
    const [mode, setPromptMode] = usePromptMode();

    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Text UNSAFE_className={styles.label}>Prompt Mode</Text>
            <ToggleButtons
                options={OPTIONS}
                selectedOption={mode}
                onOptionChange={setPromptMode}
                getOptionLabel={(option) => LABELS[option]}
            />
        </Flex>
    );
};
