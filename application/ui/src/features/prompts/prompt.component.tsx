/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Divider, Flex, Heading, View } from '@geti/ui';

import { ModelToolbar } from './model-toolbar/model-toolbar.component';
import { PromptMode } from './prompt-modes/prompt-mode.component';
import { PromptModes } from './prompt-modes/prompt-modes.component';

export const Prompt = () => {
    return (
        <View
            minWidth={'size-4600'}
            width={'100%'}
            backgroundColor={'gray-100'}
            paddingY={'size-200'}
            paddingX={'size-300'}
            height={'100%'}
        >
            <Flex direction={'column'} height={'100%'}>
                <Heading margin={0}>Prompt</Heading>
                <View padding={'size-300'} flex={1}>
                    <Flex direction={'column'} gap={'size-300'} height={'100%'}>
                        <PromptModes />

                        <Divider size={'S'} />

                        <Flex direction={'column'} gap={'size-200'}>
                            <ModelToolbar />
                            <View flex={1}>
                                <PromptMode />
                            </View>
                        </Flex>
                    </Flex>
                </View>
            </Flex>
        </View>
    );
};
