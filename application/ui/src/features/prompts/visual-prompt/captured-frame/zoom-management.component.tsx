/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Flex, Grid, Text } from '@geti/ui';
import { Add, FitScreen, Remove } from '@geti/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';
import { HOTKEYS } from 'src/features/annotator/hotkeys/hotkeys';

import { useSetZoom, useZoom } from '../../../../components/zoom/zoom.provider';

export const ZoomManagement = () => {
    const zoom = useZoom();
    const { onZoomChange, fitToScreen } = useSetZoom();

    useHotkeys(HOTKEYS.fitToScreen, fitToScreen, [fitToScreen]);

    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <Grid alignItems={'center'} columns={['size-400', 'size-450', 'size-400']} gap={'size-50'}>
                <ActionButton
                    aria-label={'Zoom in'}
                    isQuiet
                    onPress={() => onZoomChange(1)}
                    isDisabled={zoom.scale >= zoom.maxZoomIn}
                >
                    <Add />
                </ActionButton>
                <Text UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-25)' }}>
                    {(zoom.scale * 100).toFixed(1)}%
                </Text>
                <ActionButton
                    aria-label={'Zoom out'}
                    isQuiet
                    onPress={() => onZoomChange(-1)}
                    isDisabled={zoom.scale <= zoom.initialCoordinates.scale}
                >
                    <Remove />
                </ActionButton>
            </Grid>
            <ActionButton isQuiet onPress={fitToScreen} aria-label='Fit image to screen'>
                <FitScreen />
            </ActionButton>
        </Flex>
    );
};
