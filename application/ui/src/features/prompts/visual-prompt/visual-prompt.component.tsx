/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Grid } from '@geti/ui';

import { useSelectedFrame } from '../../../shared/selected-frame-provider.component';
import { CapturedFramePlaceholder } from './captured-frame/captured-frame-placeholder.component';
import { CapturedFrame, CapturedFrameProviders } from './captured-frame/captured-frame.component';
import { PromptThumbnailList } from './prompt-thumbnails/prompt-thumbnail-list/prompt-thumbnail-list.component';
import { SavePrompt } from './save-prompt/save-prompt.component';
import { useVisualPrompt } from './visual-prompt-provider.component';

export const VisualPrompt = () => {
    const { selectedFrameId } = useSelectedFrame();
    const { prompt } = useVisualPrompt();

    return (
        <Flex direction={'column'} gap={'size-300'} height={'100%'}>
            <Grid minHeight={'size-6000'} gap={'size-300'}>
                {selectedFrameId === null ? (
                    <CapturedFramePlaceholder />
                ) : (
                    <CapturedFrameProviders frameId={prompt?.frame_id ?? selectedFrameId}>
                        <CapturedFrame />
                        <SavePrompt />
                    </CapturedFrameProviders>
                )}
            </Grid>

            <PromptThumbnailList />
        </Flex>
    );
};
