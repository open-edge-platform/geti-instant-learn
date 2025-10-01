/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Flex, Text } from '@geti/ui';
import { Add, FitScreen, Remove } from '@geti/ui/icons';

export const ZoomManagement = () => {
    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <ActionButton aria-label={'Zoom in'} isQuiet>
                <Add />
            </ActionButton>
            <Text>110%</Text>
            <ActionButton aria-label={'Zoom out'} isQuiet>
                <Remove />
            </ActionButton>
            <ActionButton aria-label={'Fit image to screen'} isQuiet>
                <FitScreen />
            </ActionButton>
        </Flex>
    );
};
