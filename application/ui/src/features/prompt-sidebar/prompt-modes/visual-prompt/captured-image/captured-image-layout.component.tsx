/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, minmax, View } from '@geti/ui';

import { type CapturedImageType } from '../types';
import { AnnotatorCanvas } from './annotator/annotator-canvas.component';
import { AnnotatorProvider } from './annotator/annotator-provider.component';
import { CapturedImageActions } from './captured-image-actions.component';
import { Labels } from './labels-management/labels.component';
import { ZoomProvider } from './zoom/zoom.provider';

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
            <AnnotatorProvider size={size} image={image}>
                <ZoomProvider>
                    <View gridArea={'labels'} backgroundColor={'gray-200'} paddingX={'size-100'} paddingY={'size-50'}>
                        <Labels />
                    </View>
                    <View gridArea={'image'} backgroundColor={'gray-50'}>
                        <AnnotatorCanvas image={image} size={size} />
                    </View>
                    <View gridArea={'actions'} backgroundColor={'gray-200'} padding={'size-100'}>
                        <CapturedImageActions />
                    </View>
                </ZoomProvider>
            </AnnotatorProvider>
        </Grid>
    );
};
