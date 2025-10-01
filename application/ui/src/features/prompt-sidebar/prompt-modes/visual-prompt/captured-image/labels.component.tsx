/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Flex } from '@geti/ui';

export const Labels = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'}>
            <ActionButton isQuiet>Add label</ActionButton>
        </Flex>
    );
};
