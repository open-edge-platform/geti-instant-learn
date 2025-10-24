/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties } from 'react';

import { Grid, minmax, View } from '@geti/ui';

import { ZoomProvider } from '../../../../../components/zoom/zoom.provider';
import { useSelectedFrame } from '../../../../stream/selected-frame-provider.component';
import { CapturedFrameActions } from './captured-frame-actions.component';
import { CapturedFrame } from './captured-frame/captured-frame.component';
import { Labels } from './labels-management/labels.component';

export const CapturedFrameLayout = () => {
    const { selectedFrameId } = useSelectedFrame();

    const disabledStyles: CSSProperties = selectedFrameId === null ? { opacity: 0.5, pointerEvents: 'none' } : {};

    return (
        <Grid
            width={'100%'}
            areas={['labels', 'image', 'actions']}
            rows={[minmax('size-500', 'auto'), '1fr', 'size-500']}
            height={'100%'}
        >
            <ZoomProvider>
                <View
                    gridArea={'labels'}
                    backgroundColor={'gray-200'}
                    paddingX={'size-100'}
                    paddingY={'size-50'}
                    UNSAFE_style={disabledStyles}
                >
                    <Labels />
                </View>
                <View gridArea={'image'} backgroundColor={'gray-50'}>
                    <CapturedFrame frameId={selectedFrameId} />
                </View>
                <View
                    gridArea={'actions'}
                    backgroundColor={'gray-200'}
                    padding={'size-100'}
                    UNSAFE_style={disabledStyles}
                >
                    <CapturedFrameActions />
                </View>
            </ZoomProvider>
        </Grid>
    );
};
