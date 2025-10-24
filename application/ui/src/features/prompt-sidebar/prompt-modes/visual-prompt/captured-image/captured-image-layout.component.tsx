/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, minmax, View } from '@geti/ui';
import { ZoomTransform } from 'src/components/zoom/zoom-transform';
import { ZoomProvider } from 'src/components/zoom/zoom.provider';

import { type CapturedImageType } from '../types';
import { CapturedImageActions } from './captured-image-actions.component';
import { Labels } from './labels-management/labels.component';

interface CapturedImageLayoutProps {
    image: CapturedImageType;
}

export const CapturedImageLayout = ({ image }: CapturedImageLayoutProps) => {
    //TODO: dummy size was used, should be replaced with real image size when available
    const size = {
        width: 500,
        height: 400,
    };

    return (
        <Grid
            width={'100%'}
            areas={['labels', 'image', 'actions']}
            rows={[minmax('size-500', 'auto'), '1fr', 'size-500']}
            height={'100%'}
        >
            <ZoomProvider>
                <View gridArea={'labels'} backgroundColor={'gray-200'} paddingX={'size-100'} paddingY={'size-50'}>
                    <Labels />
                </View>
                <View gridArea={'image'} backgroundColor={'gray-50'}>
                    <ZoomTransform target={size}>
                        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                            <img
                                src={image}
                                alt={image.toString()}
                                style={{ height: '100%', width: '100%', objectFit: 'contain' }}
                            />
                        </div>
                    </ZoomTransform>
                </View>
                <View gridArea={'actions'} backgroundColor={'gray-200'} padding={'size-100'}>
                    <CapturedImageActions />
                </View>
            </ZoomProvider>
        </Grid>
    );
};
