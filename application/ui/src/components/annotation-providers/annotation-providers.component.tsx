/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import type { AnnotationType, LabelType } from '@geti-prompt/api';

import { FullScreenModeProvider } from '../../features/annotator/actions/full-screen-mode.component';
import { CanvasSettingsProvider } from '../../features/annotator/actions/settings/canvas-settings-provider.component';
import { AnnotationActionsProvider } from '../../features/annotator/providers/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../features/annotator/providers/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../../features/annotator/providers/annotator-provider.component';
import { SelectAnnotationProvider } from '../../features/annotator/providers/select-annotation-provider.component';

interface AnnotationProvidersProps {
    children: ReactNode;
    frameId: string;
    initialAnnotationsDTO?: AnnotationType[];
    labels?: LabelType[];
}

export const AnnotationProviders = ({ children, frameId, initialAnnotationsDTO, labels }: AnnotationProvidersProps) => {
    return (
        <AnnotatorProvider frameId={frameId}>
            <SelectAnnotationProvider>
                <AnnotationActionsProvider initialAnnotationsDTO={initialAnnotationsDTO} labels={labels}>
                    <AnnotationVisibilityProvider>
                        <FullScreenModeProvider>
                            <CanvasSettingsProvider>{children}</CanvasSettingsProvider>
                        </FullScreenModeProvider>
                    </AnnotationVisibilityProvider>
                </AnnotationActionsProvider>
            </SelectAnnotationProvider>
        </AnnotatorProvider>
    );
};
