/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, minmax } from '@geti/ui';

import { CapturedFrameContent } from './captured-frame-content.component';
import { CapturedFrameFullScreen } from './captured-frame-full-screen.component';

export const CapturedFrame = ({ frameId }: { frameId: string }) => {
    return (
        <Grid
            width={'100%'}
            areas={['labels', 'image', 'actions']}
            rows={[minmax('size-500', 'auto'), 'size-6000', 'size-500']}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
            }}
        >
            <CapturedFrameContent frameId={frameId} />
            <CapturedFrameFullScreen frameId={frameId} />
        </Grid>
    );
};
