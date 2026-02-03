/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex } from '@geti/ui';

import { useSelectedFrame } from '../../../shared/selected-frame-provider.component';
import { CapturedFramePlaceholder } from './captured-frame/captured-frame-placeholder.component';
import { CapturedFrame, CapturedFrameProviders } from './captured-frame/captured-frame.component';
import { PromptThumbnailList } from './prompt-thumbnails/prompt-thumbnail-list/prompt-thumbnail-list.component';
import { SavePrompt } from './save-prompt/save-prompt.component';

export const VisualPrompt = () => {
    const { selectedFrameId } = useSelectedFrame();

    return (
        <Flex direction={'column'} gap={'size-300'} height={'100%'}>
            <Flex direction={'column'} minHeight={'size-6000'} gap={'size-300'} flex={1}>
                {selectedFrameId === null ? (
                    <CapturedFramePlaceholder />
                ) : (
                    <CapturedFrameProviders frameId={selectedFrameId}>
                        <CapturedFrame frameId={selectedFrameId} />
                        <SavePrompt />
                    </CapturedFrameProviders>
                )}
            </Flex>

            <PromptThumbnailList />
        </Flex>
    );
};
