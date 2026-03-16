/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties } from 'react';

import { TextPromptType } from '@/api';
import { ActionButton, Text } from '@geti/ui';
import { Close } from '@geti/ui/icons';
import { getDistinctColorBasedOnHash } from '@geti/ui/utils';

import classes from './text-prompt-badge.module.scss';

type TextPromptBadgeProps = {
    prompt: TextPromptType;
    onDelete: (id: string) => void;
};

export const TextPromptBadge = ({ prompt, onDelete }: TextPromptBadgeProps) => {
    const color = getDistinctColorBasedOnHash(prompt.id);

    return (
        <div
            className={classes.badge}
            style={{ '--badgeBgColor': color } as CSSProperties}
            aria-label={`Text prompt: ${prompt.content}`}
        >
            <Text UNSAFE_className={classes.badgeText}>{prompt.content}</Text>
            <ActionButton
                isQuiet
                aria-label={`Delete prompt: ${prompt.content}`}
                onPress={() => onDelete(prompt.id)}
                UNSAFE_className={classes.deleteButton}
            >
                <Close />
            </ActionButton>
        </div>
    );
};
