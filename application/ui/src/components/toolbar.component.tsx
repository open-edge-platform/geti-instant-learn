/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, View } from '@geti/ui';

import { SourcesSinksConfiguration } from '../features/sources-sinks-configuration/sources-sinks-configuration.component';

export const Toolbar = () => {
    return (
        <View
            gridArea={'toolbar'}
            borderEndWidth={'thin'}
            borderTopWidth={'thin'}
            borderColor={'gray-50'}
            backgroundColor={'gray-100'}
        >
            <Flex justifyContent={'end'} alignItems={'center'} height={'100%'}>
                <SourcesSinksConfiguration />
            </Flex>
        </View>
    );
};
