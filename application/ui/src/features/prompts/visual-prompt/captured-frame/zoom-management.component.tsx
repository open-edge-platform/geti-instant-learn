/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Flex, Text } from '@geti/ui';
import { Add, FitScreen, Remove } from '@geti/ui/icons';

import { IconWrapper } from '../../../../components/icon-wrapper/icon-wrapper.component';
import { useSetZoom, useZoom } from '../../../../components/zoom/zoom.provider';

export const ZoomManagement = () => {
    const zoom = useZoom();
    const { onZoomChange, fitToScreen } = useSetZoom();

    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <ActionButton
                aria-label={'Zoom in'}
                isQuiet
                onPress={() => onZoomChange(1)}
                isDisabled={zoom.scale >= zoom.maxZoomIn}
            >
                <Add />
            </ActionButton>
            <Text> {(zoom.scale * 100).toFixed(1)}%</Text>
            <ActionButton
                aria-label={'Zoom out'}
                isQuiet
                onPress={() => onZoomChange(-1)}
                isDisabled={zoom.scale <= zoom.initialCoordinates.scale}
            >
                <Remove />
            </ActionButton>
            <ActionButton isQuiet onPress={fitToScreen} aria-label='Fit image to screen'>
                <IconWrapper>
                    <FitScreen />
                </IconWrapper>
            </ActionButton>
        </Flex>
    );
};
