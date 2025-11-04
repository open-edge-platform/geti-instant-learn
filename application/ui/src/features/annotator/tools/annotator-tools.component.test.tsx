/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { AnnotatorProvider } from '../providers/annotator-provider.component';
import { AnnotatorTools } from './annotator-tools.component';

describe('AnnotatorTools', () => {
    it('does not render SAM tool by default', async () => {
        render(
            <AnnotatorProvider frameId={'test-frame'}>
                <AnnotatorTools />
            </AnnotatorProvider>
        );

        const samButton = await screen.findByLabelText('Select sam Tool');
        expect(samButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('enables SAM tool with hotkeys', async () => {
        render(
            <AnnotatorProvider frameId={'test-frame'}>
                <AnnotatorTools />
            </AnnotatorProvider>
        );

        const samButton = await screen.findByLabelText('Select sam Tool');
        expect(samButton).toHaveAttribute('aria-pressed', 'false');

        await userEvent.keyboard('s');

        expect(samButton).toHaveAttribute('aria-pressed', 'true');
    });
});
