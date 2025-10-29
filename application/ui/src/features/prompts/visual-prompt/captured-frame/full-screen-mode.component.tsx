/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Content, Dialog, DialogTrigger, Divider, Flex, Grid, Heading, minmax } from '@geti/ui';
import { Collapse, Expand } from '@geti/ui/icons';

import { AnnotatorTools } from '../../../annotator/tools/annotator-tools.component';
import { useSelectedFrame } from '../../../stream/selected-frame-provider.component';
import { CanvasAdjustments } from './canvas-adjustments.component';
import { CapturedFrameContent, useFullScreenMode } from './captured-frame.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { ZoomManagement } from './zoom-management.component';

const CapturedFrameActionsFullScreen = () => {
    const { setIsFullScreenMode, isFullScreenMode } = useFullScreenMode();

    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'} gap={'size-50'}>
            <AnnotatorTools />

            <ToggleAnnotationsVisibility />

            <CanvasAdjustments />

            <ZoomManagement />

            <ActionButton isQuiet onPress={() => setIsFullScreenMode(!isFullScreenMode)}>
                <Collapse />
            </ActionButton>
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
    const { isFullScreenMode, setIsFullScreenMode } = useFullScreenMode();

    return (
        <DialogTrigger isOpen={isFullScreenMode} type={'fullscreenTakeover'}>
            <ActionButton isQuiet onPress={() => setIsFullScreenMode(!isFullScreenMode)}>
                <Expand />
            </ActionButton>
            <Dialog>
                <Heading>Annotate frame</Heading>
                <Divider />
                <Content>
                    <CapturedFrameFullScreen />
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
