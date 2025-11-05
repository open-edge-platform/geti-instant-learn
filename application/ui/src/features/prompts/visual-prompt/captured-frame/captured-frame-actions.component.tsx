/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useRef } from 'react';

import { DOMRefValue, Flex } from '@geti/ui';

import { Settings } from '../../../annotator/settings/settings.component';
import { FullScreenMode } from './full-screen-mode.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { UndoRedo } from './undo-redo.component';
import { ZoomManagement } from './zoom-management.component';

export const CapturedFrameActions = () => {
    const ref = useRef<DOMRefValue<HTMLDivElement>>(null);

    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'} gap={'size-50'} ref={ref}>
            <UndoRedo />

            <ToggleAnnotationsVisibility />

            <Settings ref={ref} />

            <ZoomManagement />

            <FullScreenMode />
        </Flex>
    );
};
