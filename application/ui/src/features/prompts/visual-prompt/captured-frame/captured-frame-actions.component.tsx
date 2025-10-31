/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useRef } from 'react';

import { DOMRefValue, Flex } from '@geti/ui';

import { CanvasSettings } from '../../../annotator/canvas-settings/canvas-settings.component';
import { AnnotatorTools } from '../../../annotator/tools/annotator-tools.component';
import { FullScreenMode } from './full-screen-mode.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { UndoRedo } from './undo-redo.component';
import { ZoomManagement } from './zoom-management.component';

export const CapturedFrameActions = () => {
    const ref = useRef<DOMRefValue<HTMLDivElement>>(null);

    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'} gap={'size-50'} ref={ref}>
            <AnnotatorTools />

            <UndoRedo />

            <ToggleAnnotationsVisibility />

            <CanvasSettings ref={ref} />

            <ZoomManagement />

            <FullScreenMode />
        </Flex>
    );
};
