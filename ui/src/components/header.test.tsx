/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { screen, waitForElementToBeRemoved } from '@testing-library/react';

import { Header } from './header.component';

describe('Header', () => {
    it('renders header properly', async () => {
        render(<Header />);

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));
        expect(await screen.findByText('Geti Prompt')).toBeInTheDocument();
    });
});
