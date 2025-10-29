/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, FullscreenAction, Grid, minmax } from '@geti/ui';

import { AnnotatorTools } from '../../../annotator/tools/annotator-tools.component';
import { useSelectedFrame } from '../../../stream/selected-frame-provider.component';
import { CanvasAdjustments } from './canvas-adjustments.component';
import { CapturedFrameContent } from './captured-frame.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { ZoomManagement } from './zoom-management.component';

const CapturedFrameActionsFullScreen = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'} gap={'size-50'}>
            <AnnotatorTools />

            <ToggleAnnotationsVisibility />

            <CanvasAdjustments />

            <ZoomManagement />
        </Flex>
    );
};

const CapturedFrameFullScreen = () => {
    const { selectedFrameId } = useSelectedFrame();

    // Note: In full-screen mode, we expect a frame to be selected. This is just a safeguard.
    if (selectedFrameId === null) {
        return null;
    }

    return (
        <Grid
            height={'100%'}
            width={'100%'}
            areas={['labels', 'image', 'actions']}
            rows={['size-500', minmax(0, '1fr'), 'size-500']}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
            }}
        >
            <CapturedFrameContent frameId={selectedFrameId} actions={<CapturedFrameActionsFullScreen />} />
        </Grid>
    );
};

export const FullScreenMode = () => {
    return (
        <FullscreenAction title={'Annotate frame'}>
            <CapturedFrameFullScreen />
        </FullscreenAction>
    );
};
