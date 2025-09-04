/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Header as SpectrumHeader, View } from '@geti/ui';

export const Header = () => {
    return (
        <View backgroundColor={'gray-300'} height='size-800'>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <SpectrumHeader>Geti Prompt</SpectrumHeader>
            </Flex>
        </View>
    );
};
