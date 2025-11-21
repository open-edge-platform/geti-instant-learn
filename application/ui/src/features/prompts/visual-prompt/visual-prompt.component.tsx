/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Grid, View } from '@geti/ui';

import { useSelectedFrame } from '../../../shared/selected-frame-provider.component';
import { CapturedFramePlaceholder } from './captured-frame/captured-frame-placeholder.component';
import { CapturedFrame, CapturedFrameProviders } from './captured-frame/captured-frame.component';
import { Labels } from './labels-management/labels.component';
import { PromptThumbnailList } from './prompt-thumbnails/prompt-thumbnail-list/prompt-thumbnail-list.component';
import { SavePrompt } from './save-prompt/save-prompt.component';

export const VisualPrompt = () => {
    const { selectedFrameId } = useSelectedFrame();

    return (
        <Flex direction={'column'} gap={'size-300'} height={'100%'}>
            {selectedFrameId === null ? (
                <CapturedFramePlaceholder />
            ) : (
                <CapturedFrameProviders frameId={selectedFrameId}>
                    <View gridArea={'labels'} backgroundColor={'gray-200'} paddingX={'size-100'} paddingY={'size-50'}>
                        <Labels />
                    </View>

                    <Grid minHeight={'size-6000'} gap={'size-300'}>
                        <>
                            <CapturedFrame frameId={selectedFrameId} />
                            <SavePrompt />
                        </>
                    </Grid>
                </CapturedFrameProviders>
            )}

            <PromptThumbnailList />
        </Flex>
    );
};
