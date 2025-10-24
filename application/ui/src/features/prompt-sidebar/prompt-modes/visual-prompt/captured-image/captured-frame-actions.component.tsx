/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex } from '@geti/ui';

import { CanvasAdjustments } from './canvas-adjustments.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { ZoomManagement } from './zoom-management.component';

export const CapturedFrameActions = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'} gap={'size-100'}>
            <ToggleAnnotationsVisibility />
            <CanvasAdjustments />
            <ZoomManagement />
        </Flex>
    );
};
