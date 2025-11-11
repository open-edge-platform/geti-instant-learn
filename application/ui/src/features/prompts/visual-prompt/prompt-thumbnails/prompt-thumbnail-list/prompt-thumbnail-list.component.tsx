/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VisualPromptItemType } from '@geti-prompt/api';
import { Flex, Grid, Text } from '@geti/ui';

import { useGetPrompts } from '../../api/use-get-prompts';
import { PromptThumbnail } from '../prompt-thumbnail/prompt-thumbnail.component';

export const PromptThumbnailList = () => {
    const { prompts } = useGetPrompts();

    if (prompts.length === 0) {
        return (
            <Flex marginY={'size-300'} justifyContent={'center'} alignItems={'center'}>
                <Text>No prompts available</Text>
            </Flex>
        );
    }

    return (
        <Grid columns={['1fr', '1fr']} gap={'size-100'}>
            {prompts.map((prompt) => (
                <PromptThumbnail key={prompt.id} prompt={prompt as VisualPromptItemType} />
            ))}
        </Grid>
    );
};
