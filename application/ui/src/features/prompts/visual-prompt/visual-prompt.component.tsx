/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Flex } from '@geti/ui';
import { useSelectedFrame } from 'src/features/stream/selected-frame-provider.component';

import { CapturedFrame } from './captured-frame/captured-frame.component';
import { PromptThumbnailList } from './prompt-thumbnails/prompt-thumbnail-list/prompt-thumbnail-list.component';

export const VisualPrompt = () => {
    const { selectedFrameId } = useSelectedFrame();

    const savePrompt = () => {
        // TODO: PUT /api/v1/projects/{project_id}/prompts/{prompt_id}
    };

    return (
        <Flex direction={'column'} gap={'size-300'} height={'100%'}>
            <CapturedFrame frameId={selectedFrameId} />

            <Button alignSelf={'end'} variant={'secondary'} isDisabled={selectedFrameId === null} onPress={savePrompt}>
                Save prompt
            </Button>

            <PromptThumbnailList />
        </Flex>
    );
};
