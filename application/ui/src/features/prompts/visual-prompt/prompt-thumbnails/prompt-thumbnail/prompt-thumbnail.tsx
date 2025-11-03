/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionMenu, Item, Key, View } from '@geti/ui';

import styles from './prompt-thumbnail.module.scss';

const PROMPT_OPTIONS = ['Edit', 'Delete'] as const;

interface PromptThumbnailProps {
    image: string;
}
export const PromptThumbnail = ({ image }: PromptThumbnailProps) => {
    const onAction = (option: Key) => {
        console.info(`Selected option: ${option}`);
    };

    return (
        <View UNSAFE_className={styles.promptThumbnail}>
            <img src={image} alt={image.toString()} className={styles.image} />

            <View
                position={'absolute'}
                right={'size-100'}
                top={'size-100'}
                backgroundColor={'gray-50'}
                UNSAFE_className={styles.actionMenu}
            >
                <ActionMenu isQuiet onAction={onAction} aria-label={'Prompt actions'}>
                    {PROMPT_OPTIONS.map((option) => {
                        return <Item key={option}>{option}</Item>;
                    })}
                </ActionMenu>
            </View>
        </View>
    );
};
