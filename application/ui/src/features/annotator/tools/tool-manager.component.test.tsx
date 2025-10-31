/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { screen, waitForElementToBeRemoved } from '@testing-library/react';

import { AnnotationActionsProvider } from '../providers/annotation-actions-provider.component';
import { AnnotatorProvider } from '../providers/annotator-provider.component';
import { ToolManager } from './tool-manager.component';

describe('ToolManager', () => {
    it('does not render if there is no active tool', async () => {
        render(
            <AnnotatorProvider frameId={'test-frame'}>
                <AnnotationActionsProvider>
                    <ToolManager activeTool={null} />
                </AnnotationActionsProvider>
            </AnnotatorProvider>
        );

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));

        expect(screen.queryByLabelText('SAM tool canvas')).not.toBeInTheDocument();
    });

    it('renders SAM tool correctly', async () => {
        render(
            <AnnotatorProvider frameId={'test-frame'}>
                <AnnotationActionsProvider>
                    <ToolManager activeTool={'sam'} />
                </AnnotationActionsProvider>
            </AnnotatorProvider>
        );

        expect(await screen.findByText('Processing image, please wait...')).toBeInTheDocument();
    });
});
