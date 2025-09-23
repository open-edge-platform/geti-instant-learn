/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Divider, Flex, Heading, View } from '@geti/ui';

import { PromptMode } from './prompt-modes/prompt-mode.component';
import { PromptModes } from './prompt-modes/prompt-modes.component';
import { PromptToolbar } from './prompt-modes/prompt-toolbar/prompt-toolbar.component';

export const PromptSidebar = () => {
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
                <View flex={1} padding={'size-300'}>
                    <Flex direction={'column'} height={'100%'} gap={'size-300'}>
                        <PromptModes />

                        <Divider size={'S'} />

                        <Flex flex={1} direction={'column'} gap={'size-200'}>
                            <PromptToolbar />
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
