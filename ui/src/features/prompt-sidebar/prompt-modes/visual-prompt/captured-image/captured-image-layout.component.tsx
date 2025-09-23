/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, minmax, View } from '@geti/ui';

import { type CapturedImageType } from '../types';
import { CapturedImageActions } from './captured-image-actions.component';
import { Labels } from './labels.component';

interface CapturedImageLayoutProps {
    image: CapturedImageType;
}

export const CapturedImageLayout = ({ image }: CapturedImageLayoutProps) => {
    return (
        <Grid
            width={'100%'}
            areas={['labels', 'image', 'actions']}
            rows={[minmax('size-500', 'auto'), '1fr', 'size-500']}
            height={'100%'}
        >
            <View gridArea={'labels'} backgroundColor={'gray-200'} padding={'size-100'}>
                <Labels />
            </View>
            <View gridArea={'image'} backgroundColor={'gray-50'}>
                <img
                    src={image}
                    alt={image.toString()}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
            </View>
            <View gridArea={'actions'} backgroundColor={'gray-200'} padding={'size-100'}>
                <CapturedImageActions />
            </View>
        </Grid>
    );
};
