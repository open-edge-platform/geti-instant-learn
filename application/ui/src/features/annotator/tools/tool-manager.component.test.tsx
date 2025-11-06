/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { screen } from '@testing-library/react';

import { AnnotationActionsProvider } from '../providers/annotation-actions-provider.component';
import { AnnotatorProvider } from '../providers/annotator-provider.component';
import { ToolManager } from './tool-manager.component';

describe('ToolManager', () => {
    it('renders SAM tool correctly', async () => {
        render(
            <AnnotatorProvider frameId={'test-frame'}>
                <AnnotationActionsProvider>
                    <ToolManager />
                </AnnotationActionsProvider>
            </AnnotatorProvider>
        );

        expect(await screen.findByText('Processing image, please wait...')).toBeInTheDocument();
    });
});
