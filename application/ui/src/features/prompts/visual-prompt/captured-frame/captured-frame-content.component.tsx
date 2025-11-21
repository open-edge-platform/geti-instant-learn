/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { View } from '@geti/ui';

import { ZoomProvider } from '../../../../components/zoom/zoom.provider';
import { AnnotatorActions } from '../../../annotator/actions/annotator-actions.component';
import { AnnotatorCanvasSettings } from '../../../annotator/actions/settings/annotator-canvas-settings.component';
import { AnnotatorCanvas } from '../../../annotator/annotator-canvas/annotator-canvas';

export const CapturedFrameContent = () => {
    return (
        <ZoomProvider>
            <View gridArea={'image'} backgroundColor={'gray-50'} overflow={'hidden'}>
                <AnnotatorCanvasSettings>
                    <AnnotatorCanvas />
                </AnnotatorCanvasSettings>
            </View>
            <View gridArea={'actions'} backgroundColor={'gray-200'} padding={'size-100'}>
                <AnnotatorActions />
            </View>
        </ZoomProvider>
    );
};
