/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useRef } from 'react';

import { DOMRefValue, Flex } from '@geti/ui';

import { DeleteAnnotations } from './delete-annotations.component';
import { FullScreenMode } from './full-screen-mode.component';
import { Settings } from './settings/settings.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { UndoRedo } from './undo-redo/undo-redo.component';
import { ZoomManagement } from './zoom-management.component';

export const AnnotatorActions = () => {
    const ref = useRef<DOMRefValue<HTMLDivElement>>(null);

    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'} gap={'size-50'} ref={ref}>
            <UndoRedo />

            <DeleteAnnotations />

            <ToggleAnnotationsVisibility />

            <Settings ref={ref} />

            <ZoomManagement />

            <FullScreenMode />
        </Flex>
    );
};
