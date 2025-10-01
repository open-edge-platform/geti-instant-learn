/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, View } from '@geti/ui';

import { InputOutputConfiguration } from '../features/input-output-configuration/input-output-configuration.component';

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
                <InputOutputConfiguration />
            </Flex>
        </View>
    );
};
