/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { screen, waitForElementToBeRemoved } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { http, server } from 'src/setup-test';

import { Header } from './header.component';

describe('Header', () => {
    it('renders header properly', async () => {
        server.use(
            http.get('/api/v1/projects', () => {
                return HttpResponse.json({
                    projects: [
                        {
                            id: '1',
                            name: 'Project #1',
                        },
                    ],
                });
            }),

            http.get('/api/v1/projects/{project_id}', () => {
                return HttpResponse.json({
                    id: '1',
                    name: 'Project #1',
                });
            })
        );

        render(<Header />);

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));
        // screen.debug();
        expect(await screen.findByText('Geti Prompt')).toBeInTheDocument();
    });
});
