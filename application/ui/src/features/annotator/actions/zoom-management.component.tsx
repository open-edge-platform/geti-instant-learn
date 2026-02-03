/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Flex, Grid, Text, Tooltip, TooltipTrigger } from '@geti/ui';
import { Add, FitScreen, Remove } from '@geti/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { useSetZoom, useZoom } from '../../../components/zoom/zoom.provider';
import { HOTKEYS } from './hotkeys';

export const ZoomManagement = () => {
    const zoom = useZoom();
    const { onZoomChange, fitToScreen } = useSetZoom();

    useHotkeys(HOTKEYS.fitToScreen, fitToScreen, [fitToScreen]);

    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <Grid alignItems={'center'} columns={['size-400', 'size-450', 'size-400']} gap={'size-50'}>
                <TooltipTrigger>
                    <ActionButton
                        aria-label={'Zoom out'}
                        isQuiet
                        onPress={() => onZoomChange(-1)}
                        isDisabled={zoom.scale <= zoom.initialCoordinates.scale}
                    >
                        <Remove />
                    </ActionButton>
                    <Tooltip>Zoom out</Tooltip>
                </TooltipTrigger>

                <Text
                    data-testid='zoom-level'
                    UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-25)' }}
                >
                    {(zoom.scale * 100).toFixed(1)}%
                </Text>

                <TooltipTrigger>
                    <ActionButton
                        aria-label={'Zoom in'}
                        isQuiet
                        onPress={() => onZoomChange(1)}
                        isDisabled={zoom.scale >= zoom.maxZoomIn}
                    >
                        <Add />
                    </ActionButton>
                    <Tooltip>Zoom in</Tooltip>
                </TooltipTrigger>
            </Grid>

            <TooltipTrigger>
                <ActionButton isQuiet onPress={fitToScreen} aria-label='Fit image to screen'>
                    <FitScreen />
                </ActionButton>
                <Tooltip>Fit image to screen</Tooltip>
            </TooltipTrigger>
        </Flex>
    );
};
