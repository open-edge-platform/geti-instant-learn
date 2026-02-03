/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VisualPromptItemType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
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

    const handleEdit = () => {
        setSelectedFrameId(prompt.frame_id);
        setPromptId(prompt.id);
    };

    const handleAction = (option: Key) => {
        switch (option) {
            case 'Edit':
                handleEdit();
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
        <div onClick={handleEdit} className={styles.promptThumbnail}>
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
                <ActionMenu isQuiet onAction={handleAction} aria-label={`Prompt actions ${prompt.id}`}>
                    {PROMPT_OPTIONS.map((option) => {
                        return <Item key={option}>{option}</Item>;
                    })}
                </ActionMenu>
            </View>
        </div>
    );
};
