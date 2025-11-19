/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { Button, Flex, TextArea, View } from '@geti/ui';

export const TextPrompt = () => {
    const [prompt, setPrompt] = useState<string>('Enter your text prompt here');

    const isSubmitDisabled = prompt.trim() === '';

    return (
        <View height={'100%'}>
            <View
                backgroundColor={'gray-50'}
                padding={'size-100'}
                borderRadius={'regular'}
                height={'100%'}
                maxHeight={'size-3000'}
            >
                <TextArea
                    aria-label={'Text prompt'}
                    value={prompt}
                    onChange={setPrompt}
                    width={'100%'}
                    height={'100%'}
                />
            </View>
            <Flex justifyContent={'end'}>
                <Button marginTop={'size-200'} isDisabled={isSubmitDisabled}>
                    Submit
                </Button>
            </Flex>
        </View>
    );
};
