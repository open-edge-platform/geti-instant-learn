/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { KeyboardEvent, Suspense, useState } from 'react';

import { useProjectIdentifier } from '@/hooks';
import { ActionButton, Flex, Loading, Text, TextArea } from '@geti/ui';
import { Add } from '@geti/ui/icons';

import { useCreateTextPrompt, useDeleteTextPrompt, useGetTextPrompts } from './api/use-text-prompts';
import { TextPromptBadge } from './text-prompt-badge/text-prompt-badge.component';

import styles from './text-prompt.module.scss';

const TextPromptBadgeList = () => {
    const prompts = useGetTextPrompts();
    const { projectId } = useProjectIdentifier();
    const deleteMutation = useDeleteTextPrompt();

    const handleDelete = (promptId: string) => {
        deleteMutation.mutate({
            params: { path: { project_id: projectId, prompt_id: promptId } },
        });
    };

    if (prompts.length === 0) {
        return <Text UNSAFE_className={styles.emptyState}>No text prompts yet.</Text>;
    }

    return (
        <Flex direction={'column'} gap={'size-100'}>
            {prompts.map((prompt) => (
                <TextPromptBadge key={prompt.id} prompt={prompt} onDelete={handleDelete} />
            ))}
        </Flex>
    );
};

export const TextPrompt = () => {
    const [content, setContent] = useState('');
    const { projectId } = useProjectIdentifier();
    const createMutation = useCreateTextPrompt();

    const isSubmitDisabled = content.trim() === '' || createMutation.isPending;

    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey || !e.shiftKey)) {
            e.preventDefault();
            if (!isSubmitDisabled) handleAdd();
        }
    };

    const handleAdd = () => {
        const trimmed = content.trim();

        if (!trimmed) return;

        createMutation.mutate(
            {
                body: { type: 'TEXT', content: trimmed },
                params: { path: { project_id: projectId } },
            },
            {
                onSuccess: () => setContent(''),
            }
        );
    };

    return (
        <Flex direction={'column'} gap={'size-200'}>
            <Flex gap={'size-100'} alignItems={'end'}>
                <TextArea
                    aria-label={'New text prompt'}
                    placeholder={'e.g. red car'}
                    value={content}
                    onChange={setContent}
                    onKeyDown={handleKeyDown}
                    flex={1}
                />
                <ActionButton
                    isDisabled={isSubmitDisabled}
                    onPress={handleAdd}
                    aria-label={'Add prompt'}
                >
                    <Add />
                </ActionButton>
            </Flex>
            <Suspense fallback={<Loading mode={'inline'} size={'M'} />}>
                <TextPromptBadgeList />
            </Suspense>
        </Flex>
    );
};
