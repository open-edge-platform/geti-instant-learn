/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionMenu, Item, Key, View } from '@geti/ui';
import { useSelectedFrame } from 'src/features/stream/selected-frame-provider.component';

import { useVisualPrompt } from '../../visual-prompt-provider.component';

import styles from './prompt-thumbnail.module.scss';

const PROMPT_OPTIONS = ['Edit', 'Delete'] as const;

interface PromptThumbnailProps {
    image: { url: string; frameId: string; promptId: string };
}
export const PromptThumbnail = ({ image }: PromptThumbnailProps) => {
    const { setSelectedFrameId } = useSelectedFrame();
    const { setPromptId } = useVisualPrompt();

    const onAction = (option: Key) => {
        switch (option) {
            case 'Edit':
                setSelectedFrameId(image.frameId);
                setPromptId(image.promptId);
                break;
            case 'Delete':
                // TODO: DELETE /api/v1/projects/{project_id}/prompts/{prompt_id}
                break;
        }
    };

    return (
        <View UNSAFE_className={styles.promptThumbnail}>
            <img src={image.url} alt={image.frameId} className={styles.image} />

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
