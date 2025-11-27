/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VisualPromptItemType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { ActionMenu, Item, Key, View } from '@geti/ui';

import { useSelectedFrame } from '../../../../../shared/selected-frame-provider.component';
import { useDeletePrompt } from '../../api/use-delete-prompt';
import { useVisualPrompt } from '../../visual-prompt-provider.component';

import styles from './prompt-thumbnail.module.scss';

const PROMPT_OPTIONS = ['Edit', 'Delete'] as const;

interface PromptThumbnailProps {
    prompt: VisualPromptItemType;
}

export const PromptThumbnail = ({ prompt }: PromptThumbnailProps) => {
    const { projectId } = useProjectIdentifier();
    const { setSelectedFrameId } = useSelectedFrame();
    const { setPromptId } = useVisualPrompt();
    const deletePromptMutation = useDeletePrompt();

    const onAction = (option: Key) => {
        switch (option) {
            case 'Edit':
                setSelectedFrameId(prompt.frame_id);
                setPromptId(prompt.id);
                break;
            case 'Delete':
                deletePromptMutation.mutate({
                    params: {
                        path: {
                            project_id: projectId,
                            prompt_id: prompt.id,
                        },
                    },
                });
                break;
        }
    };

    return (
        <View UNSAFE_className={styles.promptThumbnail}>
            <img
                aria-label={`prompt thumbnail ${prompt.id}`}
                src={prompt.thumbnail}
                alt={prompt.frame_id}
                className={styles.image}
            />

            <View
                position={'absolute'}
                right={'size-100'}
                top={'size-100'}
                backgroundColor={'gray-50'}
                UNSAFE_className={styles.actionMenu}
            >
                <ActionMenu isQuiet onAction={onAction} aria-label={`Prompt actions ${prompt.id}`}>
                    {PROMPT_OPTIONS.map((option) => {
                        return <Item key={option}>{option}</Item>;
                    })}
                </ActionMenu>
            </View>
        </View>
    );
};
