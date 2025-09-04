/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { screen } from '@testing-library/react';

import { Header } from './header.component';

describe('Header', () => {
    it('renders header properly', () => {
        render(<Header />);

        expect(screen.getByText('Geti Prompt')).toBeInTheDocument();
    });
});
