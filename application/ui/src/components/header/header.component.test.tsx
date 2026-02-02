/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@/test-utils';
import { screen } from '@testing-library/react';

import { Header } from './header.component';

describe('Header', () => {
    it('renders header properly', async () => {
        render(
            <Header homeLink=''>
                <div>Here we are</div>
            </Header>
        );

        expect(await screen.findByText('Geti Prompt')).toBeInTheDocument();
        expect(await screen.findByText('Here we are')).toBeInTheDocument();
    });
});
