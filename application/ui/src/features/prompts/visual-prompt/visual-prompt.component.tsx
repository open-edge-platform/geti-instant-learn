/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Flex, View } from '@geti/ui';

import { CapturedFrame } from './captured-frame/captured-frame.component';
import { ReferencedImages } from './referenced-images.component';

export const VisualPrompt = () => {
    return (
        <Flex height={'100%'} direction={'column'} gap={'size-300'}>
            <View flex={1}>
                <CapturedFrame />
            </View>
            <Button alignSelf={'end'} variant={'secondary'}>
                Save prompt
            </Button>
            <ReferencedImages />
        </Flex>
    );
};
