/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { usePromptMode, type PromptMode } from '@/hooks';
import { Flex, Text, ToggleButtons } from '@geti/ui';

import styles from './prompt-modes.module.scss';

const LABELS: Record<PromptMode, string> = {
    VISUAL: 'Visual Prompt',
    TEXT: 'Text Prompt',
};

const OPTIONS = Object.values(LABELS);

const LABEL_TO_MODE: Record<string, PromptMode> = {
    'Visual Prompt': 'VISUAL',
    'Text Prompt': 'TEXT',
};

export const PromptModes = () => {
    const [mode, setPromptMode] = usePromptMode();

    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Text UNSAFE_className={styles.label}>Prompt Mode</Text>
            <ToggleButtons
                options={OPTIONS}
                selectedOption={LABELS[mode]}
                onOptionChange={(label) => setPromptMode(LABEL_TO_MODE[label])}
            />
        </Flex>
    );
};
