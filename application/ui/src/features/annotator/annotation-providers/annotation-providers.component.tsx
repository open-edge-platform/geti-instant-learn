/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import type { AnnotationType, LabelType } from '@geti-prompt/api';

import { CanvasSettingsProvider } from '../actions/settings/canvas-settings-provider.component';
import { AnnotationActionsProvider } from '../providers/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../providers/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../providers/annotator-provider.component';
import { SelectAnnotationProvider } from '../providers/select-annotation-provider.component';

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
                        <CanvasSettingsProvider>{children}</CanvasSettingsProvider>
                    </AnnotationVisibilityProvider>
                </AnnotationActionsProvider>
            </SelectAnnotationProvider>
        </AnnotatorProvider>
    );
};
