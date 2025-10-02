/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { View } from '@geti/ui';

import { NoMediaPlaceholder } from './no-media-placeholder/no-media-placeholder.component';

const NoSourcePlaceholder = () => {
    return (
        <View paddingX={'size-800'} paddingY={'size-1000'} height={'100%'}>
            <NoMediaPlaceholder title={'Setup your input source'} />
        </View>
    );
};

export const MainContent = () => {
    return (
        <View gridArea={'main'} backgroundColor={'gray-50'}>
            <NoSourcePlaceholder />
        </View>
    );
};
