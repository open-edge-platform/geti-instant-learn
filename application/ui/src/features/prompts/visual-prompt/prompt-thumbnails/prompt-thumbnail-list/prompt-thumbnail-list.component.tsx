/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Grid, minmax, repeat, Text } from '@geti/ui';

import { useGetPrompts } from '../../api/use-get-prompts';
import { PromptThumbnail } from '../prompt-thumbnail/prompt-thumbnail.component';

export const PromptThumbnailList = () => {
    const prompts = useGetPrompts();

    const visualPrompts = prompts.filter((prompt) => prompt.type === 'VISUAL');

    if (visualPrompts.length === 0) {
        return (
            <Flex marginY={'size-300'} justifyContent={'center'} alignItems={'center'}>
                <Text>No prompts available</Text>
            </Flex>
        );
    }

    return (
        <Grid columns={repeat('auto-fill', minmax('size-2400', '1fr'))} gap={'size-100'}>
            {visualPrompts.map((prompt) => (
                <PromptThumbnail key={prompt.id} prompt={prompt} />
            ))}
        </Grid>
    );
};
